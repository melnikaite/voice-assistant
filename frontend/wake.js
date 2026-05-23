// Browser-side wake-word detector — JS port of openWakeWord's streaming
// pipeline (three ONNX models: melspectrogram → speech embedding → wake).
// Constants taken from openwakeword/utils.py AudioFeatures.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.mjs';

const SAMPLE_RATE = 16000;
const RAW_BUFFER_MAX = SAMPLE_RATE * 10; // 160000
const MELSPEC_PAD_SAMPLES = 160 * 3; // 480
const SAMPLES_PER_CHUNK = 1280; // 80ms — openwakeword minimum
const MEL_BINS = 32;
const MEL_BUFFER_MAX_FRAMES = 970; // ~10s
const MEL_INIT_FRAMES = 76;
const EMBEDDING_WINDOW = 76; // mel frames per embedding
const EMBEDDING_STEP = 8; // mel frames stride between embeddings (per 1280-sample audio step)
const EMBEDDING_DIM = 96;
const FEATURE_BUFFER_MAX_FRAMES = 120;
const WAKE_INPUT_FRAMES = 16;

// Cooldown after a wake trigger so we don't re-fire on the same utterance.
// 16 embeddings × ~80ms ≈ 1.3s of "silence" before next detection becomes possible.
const POST_WAKE_COOLDOWN_CHUNKS = 16;

export class BrowserWakeDetector {
  constructor({ modelUrls, threshold = 0.5, onWake, onScore = null, logger = () => {} }) {
    this.modelUrls = modelUrls;
    this.threshold = threshold;
    this.onWake = onWake;
    this.onScore = onScore;
    this.log = logger;

    this.melSession = null;
    this.embSession = null;
    this.wkSession = null;

    this.melInName = 'input';
    this.embInName = 'input_1';
    this.wkInName = 'x.1';
    this.melOutName = null;
    this.embOutName = null;
    this.wkOutName = null;

    this.rawBuf = new Int16Array(0);
    this.melBuf = null;
    this.melFrames = 0;
    this.featBuf = null;
    this.featFrames = 0;
    this.accSamples = 0;
    this.cooldown = 0;

    this._busy = false;
    this._dropped = 0;
  }

  async init() {
    // Run on WASM (CPU). Disable threads — no COOP/COEP on plain http://.
    ort.env.wasm.numThreads = 1;
    // ort.env.wasm.proxy = false;  // simpler; we control concurrency ourselves

    [this.melSession, this.embSession, this.wkSession] = await Promise.all([
      ort.InferenceSession.create(this.modelUrls.melspec, { executionProviders: ['wasm'] }),
      ort.InferenceSession.create(this.modelUrls.embedding, { executionProviders: ['wasm'] }),
      ort.InferenceSession.create(this.modelUrls.wake, { executionProviders: ['wasm'] }),
    ]);
    this.melOutName = this.melSession.outputNames[0];
    this.embOutName = this.embSession.outputNames[0];
    this.wkOutName = this.wkSession.outputNames[0];
    this.log(
      `wake models loaded (mel.out=${this.melOutName}, emb.out=${this.embOutName}, wk.out=${this.wkOutName})`
    );

    // Mel buffer pre-filled with ones (76, 32), as in openwakeword
    this.melBuf = new Float32Array(MEL_BUFFER_MAX_FRAMES * MEL_BINS);
    this.melBuf.fill(1, 0, MEL_INIT_FRAMES * MEL_BINS);
    this.melFrames = MEL_INIT_FRAMES;

    // Feature buffer starts zeroed — will warm up after ~1.5s of input
    this.featBuf = new Float32Array(FEATURE_BUFFER_MAX_FRAMES * EMBEDDING_DIM);
    this.featFrames = WAKE_INPUT_FRAMES; // pretend we have a 16-frame history of zeros
  }

  /**
   * Feed an Int16Array of PCM samples (16kHz mono). Returns the latest wake
   * score, or null if not enough audio yet. Internally async — if a previous
   * call is still running, the new audio is buffered and processed later.
   */
  async feed(pcmInt16) {
    // Append to raw buffer
    const merged = new Int16Array(this.rawBuf.length + pcmInt16.length);
    merged.set(this.rawBuf, 0);
    merged.set(pcmInt16, this.rawBuf.length);
    this.rawBuf =
      merged.length > RAW_BUFFER_MAX ? merged.subarray(merged.length - RAW_BUFFER_MAX) : merged;
    this.accSamples += pcmInt16.length;

    if (this._busy) {
      // ONNX runs are async — if previous still in flight, just accumulate.
      this._dropped += 1;
      return null;
    }
    if (this.accSamples < SAMPLES_PER_CHUNK) return null;

    this._busy = true;
    try {
      return await this._processAccumulated();
    } finally {
      this._busy = false;
    }
  }

  async _processAccumulated() {
    // 1) Run melspectrogram on (accSamples + padding) most recent samples
    const padded = Math.min(this.rawBuf.length, this.accSamples + MELSPEC_PAD_SAMPLES);
    const slice = this.rawBuf.subarray(this.rawBuf.length - padded);
    const melInput = new Float32Array(slice.length);
    for (let i = 0; i < slice.length; i++) melInput[i] = slice[i]; // Int16 -> Float32 (keep magnitude)
    const melTensor = new ort.Tensor('float32', melInput, [1, melInput.length]);
    const melResult = await this.melSession.run({ [this.melInName]: melTensor });
    const melOut = melResult[this.melOutName];
    const dims = melOut.dims; // [1, 1, T, 32]
    const T = dims[2];
    // Apply openwakeword transform: x/10 + 2
    const newMel = new Float32Array(T * MEL_BINS);
    const src = melOut.data;
    for (let i = 0; i < newMel.length; i++) newMel[i] = src[i] / 10 + 2;

    // Append to mel ring buffer (in-place left-shift if needed)
    this._appendMel(newMel, T);

    // 2) Compute embeddings: one per "step" of 8 mel frames per 1280-sample audio chunk.
    const nChunks = Math.floor(this.accSamples / SAMPLES_PER_CHUNK); // ≥ 1
    // Python: for i in [nChunks-1 ... 0]: ndx = -8*i (or len if i==0)
    for (let i = nChunks - 1; i >= 0; i--) {
      let endNdx;
      if (i === 0) endNdx = this.melFrames;
      else endNdx = this.melFrames - EMBEDDING_STEP * i;
      const startNdx = endNdx - EMBEDDING_WINDOW;
      if (startNdx < 0) continue; // not enough mel history yet
      const window = this.melBuf.subarray(startNdx * MEL_BINS, endNdx * MEL_BINS);
      const embTensor = new ort.Tensor('float32', new Float32Array(window), [
        1,
        EMBEDDING_WINDOW,
        MEL_BINS,
        1,
      ]);
      const embResult = await this.embSession.run({ [this.embInName]: embTensor });
      const emb = embResult[this.embOutName].data; // Float32Array, length 96
      this._appendFeature(emb);
    }

    this.accSamples = 0;

    // 3) Run wake model on the last WAKE_INPUT_FRAMES embeddings
    if (this.cooldown > 0) {
      this.cooldown -= nChunks;
      return 0;
    }
    if (this.featFrames < WAKE_INPUT_FRAMES) return 0;
    const wkInput = new Float32Array(WAKE_INPUT_FRAMES * EMBEDDING_DIM);
    const fb = this.featBuf;
    const offset = (this.featFrames - WAKE_INPUT_FRAMES) * EMBEDDING_DIM;
    for (let i = 0; i < wkInput.length; i++) wkInput[i] = fb[offset + i];
    const wkTensor = new ort.Tensor('float32', wkInput, [1, WAKE_INPUT_FRAMES, EMBEDDING_DIM]);
    const wkResult = await this.wkSession.run({ [this.wkInName]: wkTensor });
    const score = wkResult[this.wkOutName].data[0];

    if (this.onScore) this.onScore(score);

    if (score >= this.threshold) {
      this.log(`wake triggered (score=${score.toFixed(3)})`);
      this.cooldown = POST_WAKE_COOLDOWN_CHUNKS;
      this.onWake(score);
    }
    return score;
  }

  _appendMel(newMel, newFrames) {
    const total = this.melFrames + newFrames;
    if (total > MEL_BUFFER_MAX_FRAMES) {
      // Shift left to drop oldest, keep last (MEL_BUFFER_MAX_FRAMES - newFrames) frames
      const keep = MEL_BUFFER_MAX_FRAMES - newFrames;
      this.melBuf.copyWithin(0, (this.melFrames - keep) * MEL_BINS, this.melFrames * MEL_BINS);
      this.melBuf.set(newMel, keep * MEL_BINS);
      this.melFrames = MEL_BUFFER_MAX_FRAMES;
    } else {
      this.melBuf.set(newMel, this.melFrames * MEL_BINS);
      this.melFrames = total;
    }
  }

  _appendFeature(emb) {
    if (this.featFrames >= FEATURE_BUFFER_MAX_FRAMES) {
      this.featBuf.copyWithin(0, EMBEDDING_DIM, this.featFrames * EMBEDDING_DIM);
      this.featBuf.set(emb, (this.featFrames - 1) * EMBEDDING_DIM);
    } else {
      this.featBuf.set(emb, this.featFrames * EMBEDDING_DIM);
      this.featFrames += 1;
    }
  }

  /** Hard reset (call after a complete utterance is sent). */
  reset() {
    this.melBuf.fill(0);
    this.melBuf.fill(1, 0, MEL_INIT_FRAMES * MEL_BINS);
    this.melFrames = MEL_INIT_FRAMES;
    this.featBuf.fill(0);
    this.featFrames = WAKE_INPUT_FRAMES;
    this.accSamples = 0;
    this.rawBuf = new Int16Array(0);
    this.cooldown = 0;
  }
}
