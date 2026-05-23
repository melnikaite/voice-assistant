// AudioWorklet: captures Float32 mono samples from microphone, converts to
// Int16 PCM, and emits 80ms chunks (1280 samples @ 16kHz) to the main thread.

class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.target = 1280;
    this.buffer = new Int16Array(this.target);
    this.fill = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this.buffer[this.fill++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.fill >= this.target) {
        // Transfer the underlying buffer to avoid copying
        const out = this.buffer.buffer;
        this.port.postMessage(out, [out]);
        this.buffer = new Int16Array(this.target);
        this.fill = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-recorder', PCMRecorder);
