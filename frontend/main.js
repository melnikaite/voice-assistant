import { BrowserWakeDetector } from './wake.js';
import { applyI18n, currentLang, setLang, syncFromSettings, t as i18n } from './i18n.js';
import { initItems, destroyItems } from './items.js';

// Translate everything tagged with data-i18n right at boot so the user
// never sees the English fallback flash.  Re-runs when the user picks
// a new language via the Settings panel.
applyI18n();

// ---------------------------------------------------------------------------
// Persistent client identity — survives reload + restart via localStorage.
// Passed to the server as ?client_id=… so it can restore conversation
// history and find semantically similar past turns from long-term memory.
// ---------------------------------------------------------------------------
const clientId = (() => {
  const KEY = 'va_client_id';
  let id = localStorage.getItem(KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(KEY, id);
  }
  return id;
})();

// DOM refs
const stateEl = document.querySelector('#state');
const stateCardEl = document.querySelector('#state-card');
const stateLabelEl = document.querySelector('#state-label');
const stateSubEl = document.querySelector('#state-sub');
const transcriptEl = document.querySelector('#transcript');
const responseEl = document.querySelector('#response');
const toolEl = document.querySelector('#tool');
const startBtn = document.querySelector('#start');
const stopBtn = document.querySelector('#stop');
const cancelBtn = document.querySelector('#cancel');
const pttBtn = document.querySelector('#ptt');
const logEl = document.querySelector('#log');
const sampleRateEl = document.querySelector('#sample-rate');
const wakeScoreEl = document.querySelector('#wake-score');
const muteSoundsEl = document.querySelector('#mute-sounds');
const remoteTtsEl = document.querySelector('#remote-tts');
// Image attach DOM refs — Sprint 6.  Picking/dropping an image stages
// it for the NEXT voice utterance, which is then routed to Claude
// vision on the server.  Refs are nullable to keep the rest of the
// script working on older HTML snapshots.
const attachBtnEl = document.querySelector('#attach-btn');
const attachInputEl = document.querySelector('#attach-input');
const attachPreviewEl = document.querySelector('#attach-preview');
const attachThumbEl = document.querySelector('#attach-thumb');
const attachMetaEl = document.querySelector('#attach-meta');
const attachClearEl = document.querySelector('#attach-clear');

// ---- Sound cues ----
let soundCtx = null;
function ensureSoundCtx() {
  if (!soundCtx) soundCtx = new AudioContext();
  return soundCtx;
}
function beep({ freq = 660, ms = 120, gain = 0.07, type = 'sine', slide = 0 }) {
  if (muteSoundsEl?.checked) return;
  const ctx = ensureSoundCtx();
  const t0 = ctx.currentTime;
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.connect(g);
  g.connect(ctx.destination);
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t0);
  if (slide) osc.frequency.exponentialRampToValueAtTime(freq * slide, t0 + ms / 1000);
  g.gain.setValueAtTime(0, t0);
  g.gain.linearRampToValueAtTime(gain, t0 + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + ms / 1000);
  osc.start(t0);
  osc.stop(t0 + ms / 1000 + 0.05);
}
const SOUNDS = {
  wake: () => beep({ freq: 880, ms: 100, type: 'sine' }),
  followup_open: () => beep({ freq: 660, ms: 90, type: 'sine' }),
  session_end: () => beep({ freq: 440, ms: 250, type: 'sine', slide: 0.5 }),
  ready: () => beep({ freq: 520, ms: 80, gain: 0.05 }),
};

// Map state values to i18n key prefixes.  Labels are resolved lazily
// via t() at render time so they update when the language changes.
const STATE_LABEL_KEYS = {
  listening_wake: 'state.listening_wake',
  recording:      'state.recording',
  // `processing.sub` is overridden on the fly by `progress` messages
  // from the server — see PROGRESS_LABELS below.  The placeholder
  // here shows briefly only when the first `progress` event hasn't
  // landed yet.
  processing:     'state.processing',
  speaking:       'state.speaking',
  continuation:   'state.continuation',
  loading_models: 'state.loading_models',
  '—': null,  // special: falls back to state.idle / state.idle_hint
};

// Pipeline-stage messages from the server → i18n key for human-readable
// phrase shown in the yellow PROCESSING card's sub-text.  Keep phrases
// short, active-voice, present-tense — the user reads them while the
// dot blinks.
//
// Two kinds of step exist:
//   1. Pipeline-level   (asr, agent) — emitted by the orchestrator
//      before any tool runs.
//   2. Tool-level       — emitted by individual tools at meaningful
//      points.  Each tool can have several (web_search has four).
//      Tools that emit nothing fall back to the generic "tool" step
//      with the tool's name as detail (see agent.py).
//
// When you add a new tool: pick a step name, register a key here
// and a CATALOG entry in i18n.js.
const PROGRESS_LABEL_KEYS = {
  // pipeline-level
  asr:              'progress.asr',
  agent:            'progress.agent',
  tool:             'progress.tool',             // fallback; detail = tool name
  // web_search
  localize:         'progress.localize',
  search:           'progress.search',
  fetch:            'progress.fetch',
  summarize:        'progress.summarize',
  // general_answer
  thinking:         'progress.thinking',
  // reminders
  create_reminder:  'progress.create_reminder',
  cancel_reminder:  'progress.cancel_reminder',
  list_reminders:   'progress.list_reminders',
};

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
//
// Why no SpeechSynthesis state any more:
//   In the WebRTC architecture the assistant's voice arrives as a *remote
//   audio track* from the server (Piper TTS → RTC track → <audio> in the
//   browser).  That route runs through the browser's WebRTC voice engine,
//   which means AEC3 sees what's being played and cancels speaker→mic
//   feedback automatically.  No more half-duplex mute, no Web Speech API,
//   no client-side voice/lang detection — the server picks everything.
let ws = null;             // signalling channel (SDP, ICE, JSON state events)
let pc = null;             // RTCPeerConnection — audio I/O
let mediaStream = null;    // mic capture (getUserMedia)
let audioCtx = null;       // AudioContext used to tap mic for wake detection
let workletNode = null;
let micSource = null;
let wakeDetector = null;

let lastResponseText = '';      // for the (unused-server-side) UI hint
let serverState = 'listening_wake';

function log(msg) {
  const t = new Date().toLocaleTimeString();
  logEl.textContent = `[${t}] ${msg}\n` + logEl.textContent;
}

let prevState = '—';
function setState(value) {
  stateEl.textContent = value;
  stateCardEl.dataset.state = value;
  const keyPrefix = STATE_LABEL_KEYS[value];
  if (keyPrefix) {
    stateLabelEl.textContent = i18n(`${keyPrefix}.label`);
    stateSubEl.textContent   = i18n(`${keyPrefix}.sub`);
  } else if (value === '—') {
    stateLabelEl.textContent = i18n('state.idle');
    stateSubEl.textContent   = i18n('state.idle_hint');
  } else {
    stateLabelEl.textContent = value;
    stateSubEl.textContent   = '';
  }
  if (value !== prevState) {
    if (value === 'recording' && prevState !== 'continuation') {
      SOUNDS.wake();
    } else if (value === 'continuation') {
      SOUNDS.followup_open();
    } else if (value === 'listening_wake' && prevState === 'continuation') {
      SOUNDS.session_end();
    }
    prevState = value;
  }
}

// ---------------------------------------------------------------------------
// Browser notifications (kept for tab-not-focused push delivery)
// ---------------------------------------------------------------------------
function _notify(text) {
  if (!('Notification' in window)) return;
  if (Notification.permission === 'granted') {
    new Notification('🎙 Voice Assistant', { body: text, icon: '' });
  }
}

// ---------------------------------------------------------------------------
// start() — the whole rig: mic, AudioContext (wake detection), WebRTC peer,
// WebSocket signalling.  Order matters: we need a MediaStream before
// negotiating, and we need the WS open before sending an offer.
// ---------------------------------------------------------------------------
// Wake-word config — fetched from /api/config at start().  Defaults
// here mirror the server-side defaults so a missing/failed /api/config
// call doesn't break the page; they just won't be overridable via env
// in that fallback.
let _wakeConfig = { name: 'hey_jarvis_v0.1', threshold: 0.5 };

async function loadWakeConfig() {
  try {
    const r = await fetch('/api/config');
    const data = await r.json();
    const wake = data?.wake_word;
    if (wake && typeof wake.name === 'string' && wake.name) {
      _wakeConfig.name = wake.name;
    }
    if (wake && typeof wake.threshold === 'number' && wake.threshold > 0) {
      _wakeConfig.threshold = wake.threshold;
    }
    log(`wake config: name=${_wakeConfig.name} threshold=${_wakeConfig.threshold}`);
  } catch (e) {
    log(`/api/config failed (${e.message}) — using defaults`);
  }
}

async function start() {
  startBtn.disabled = true;
  // Notification permission is requested on successful login (see
  // loadAuthState → _requestNotificationPermission), not on start —
  // we don't notify-for-anonymous so there's no reason to nag the
  // user before they have a profile.
  try {
    setState('loading_models');
    log('Loading wake-word models...');
    await loadWakeConfig();
    wakeDetector = new BrowserWakeDetector({
      modelUrls: {
        melspec: './models/melspectrogram.onnx',
        embedding: './models/embedding_model.onnx',
        wake: `./models/${_wakeConfig.name}.onnx`,
      },
      threshold: _wakeConfig.threshold,
      onWake: (score) => {
        log(`local wake! score=${score.toFixed(3)}`);
        // No need to stop TTS client-side — the server cancels it itself
        // on `wake_detected` in SPEAKING state.  Browser AEC subtracts
        // the residual playback echo from the new mic stream.
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'wake_detected', score }));
        }
      },
      onScore: (score) => {
        if (wakeScoreEl) wakeScoreEl.textContent = score.toFixed(3);
      },
      logger: log,
    });
    await wakeDetector.init();
    log('Wake models ready');
    SOUNDS.ready();

    // Capture mic.  echoCancellation: true is the entry ticket — it makes
    // the browser AEC algorithm look at any remote audio playing
    // alongside (in our case the TTS coming back over the RTC peer) and
    // subtract it from this stream before consumers (PC sender + our
    // AudioWorklet) ever see it.  autoGainControl off because it
    // introduces level pumping that confuses VAD.
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: 16000,
      },
    });

    // AudioContext consumes the same MediaStream as the WebRTC sender,
    // exposing the cleaned (post-AEC) PCM to our wake detector.  No PCM
    // gets sent over the WS any more — audio transport is WebRTC only.
    audioCtx = new AudioContext({ sampleRate: 16000 });
    sampleRateEl.textContent = `${audioCtx.sampleRate} Hz`;
    if (audioCtx.sampleRate !== 16000) {
      log(`⚠ expected 16000Hz, got ${audioCtx.sampleRate} — wake detection broken`);
    }
    await audioCtx.audioWorklet.addModule('./worklet.js');
    micSource = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, 'pcm-recorder');
    workletNode.port.onmessage = (e) => {
      // The worklet emits Int16 mono 16 kHz frames; we only feed the
      // wake detector now.  Mic → server goes over the RTC peer.
      if (wakeDetector) wakeDetector.feed(new Int16Array(e.data));
    };
    micSource.connect(workletNode);

    // WebSocket — signalling only.  Open BEFORE creating the offer
    // because we need to send the offer over it once SDP is ready.
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(
      `${proto}//${location.host}/ws?client_id=${encodeURIComponent(clientId)}`
    );
    ws.onopen = async () => {
      log('WS opened');
      stopBtn.disabled = false;
      cancelBtn.disabled = false;
      pttBtn.disabled = false;
      try {
        await startWebRtc();
      } catch (err) {
        log(`webrtc start failed: ${err.message}`);
        console.error(err);
      }
    };
    ws.onclose = () => {
      log('WS closed');
      cleanup();
    };
    ws.onerror = () => log('WS error');
    ws.onmessage = onWsMessage;
  } catch (e) {
    log(`Start failed: ${e.message}`);
    console.error(e);
    cleanup();
  }
}

// ---------------------------------------------------------------------------
// WebRTC negotiation
// ---------------------------------------------------------------------------
async function startWebRtc() {
  pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
  });

  // Receive the assistant's voice.  Server adds its TTS track BEFORE
  // creating the answer, so the answer SDP includes the inbound m= line
  // and `ontrack` fires with the audio MediaStream.
  pc.addTransceiver('audio', { direction: 'recvonly' });

  // Send the mic.  Adding the track also schedules an m= line on our
  // offer; the server matches it with its mic-track listener.
  for (const track of mediaStream.getAudioTracks()) {
    pc.addTrack(track, mediaStream);
  }

  pc.ontrack = (ev) => {
    log(`rtc: inbound track kind=${ev.track.kind} (${ev.streams.length} streams)`);
    if (ev.track.kind === 'audio' && remoteTtsEl) {
      // Critical for AEC: assigning the remote track to an <audio>
      // element routes playback through the browser's WebRTC voice
      // engine.  AEC3 inside that engine then uses this exact PCM as
      // the reference signal it subtracts from the mic.  If we played
      // via AudioContext.destination instead the AEC would not see it.
      remoteTtsEl.srcObject = ev.streams[0] || new MediaStream([ev.track]);
      remoteTtsEl.play().catch((e) => log(`remote audio play failed: ${e.message}`));
    }
  };

  pc.onicecandidate = (ev) => {
    if (!ev.candidate) return;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'webrtc_ice',
          candidate: {
            candidate: ev.candidate.candidate,
            sdpMid: ev.candidate.sdpMid,
            sdpMLineIndex: ev.candidate.sdpMLineIndex,
          },
        })
      );
    }
  };

  pc.oniceconnectionstatechange = () => {
    log(`rtc ICE: ${pc.iceConnectionState}`);
  };
  pc.onconnectionstatechange = () => {
    log(`rtc state: ${pc.connectionState}`);
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws.send(
    JSON.stringify({
      type: 'webrtc_offer',
      sdp: pc.localDescription.sdp,
      sdp_type: pc.localDescription.type,
    })
  );
  log('rtc: sent offer');
}

// ---------------------------------------------------------------------------
// WS message handler — signalling answers + state events.
// ---------------------------------------------------------------------------
async function onWsMessage(ev) {
  if (typeof ev.data !== 'string') return;
  const msg = JSON.parse(ev.data);
  switch (msg.type) {
    case 'ready': {
      const turns = msg.history_turns ?? 0;
      log(
        `ready (session=${msg.session_id}${turns ? `, history=${turns} turns restored` : ''})`
      );
      setState(msg.state);
      serverState = msg.state;
      break;
    }
    case 'webrtc_answer':
      try {
        await pc.setRemoteDescription({
          type: msg.sdp_type || 'answer',
          sdp: msg.sdp,
        });
        log('rtc: applied answer');
      } catch (e) {
        log(`rtc: setRemoteDescription failed: ${e.message}`);
      }
      break;
    case 'webrtc_ice':
      try {
        await pc.addIceCandidate(msg.candidate || null);
      } catch (e) {
        log(`rtc: addIceCandidate failed: ${e.message}`);
      }
      break;
    case 'state':
      serverState = msg.value;
      setState(msg.value);
      if (msg.value === 'continuation' && msg.timeout_s) {
        // Server now controls TTS, so by the time we hit CONTINUATION
        // the audio has already drained — start the countdown now.
        startContinuationCountdown(msg.timeout_s);
      } else if (msg.value !== 'continuation') {
        clearContinuationCountdown();
      }
      if (msg.value === 'listening_wake' && wakeDetector) {
        wakeDetector.reset();
      }
      break;
    case 'progress': {
      // Pipeline-stage detail surfaced in the yellow PROCESSING card.
      // Map the step token to a friendly phrase; ignore unknown ones
      // rather than show "step:foo" to the user.
      const progressKey = PROGRESS_LABEL_KEYS[msg.step];
      if (progressKey && stateCardEl.dataset.state === 'processing') {
        // detail is optional context (e.g. target language for localize)
        const suffix = msg.detail ? ` (${msg.detail})` : '';
        stateSubEl.textContent = i18n(progressKey) + suffix + ' …';
      }
      log(`progress: ${msg.step}${msg.detail ? ` (${msg.detail})` : ''}`);
      break;
    }
    case 'wake_ack':
      log(`server ack wake${msg.via ? ` (via ${msg.via})` : ''}`);
      transcriptEl.textContent = '…';
      responseEl.textContent = '';
      toolEl.textContent = '';
      break;
    case 'transcript': {
      const speaker = msg.speaker ? `[${msg.speaker}] ` : '';
      transcriptEl.textContent = msg.text ? speaker + msg.text : i18n('transcript.empty');
      log(`transcript (asr=${msg.asr_ms}ms${msg.speaker ? `, speaker=${msg.speaker}` : ''}): ${msg.text}`);
      break;
    }
    case 'tool_called': {
      // Multi-agent: when a tool ran against a specific desktop-agent,
      // the server stamps target_agent into the WS payload.  We render
      // it inline («computer_use · @macbook») and flash the matching
      // row in the agents panel so the user sees which device acted.
      const suffix = msg.target_agent ? ` · @${msg.target_agent}` : '';
      toolEl.textContent = msg.name
        ? `${msg.name}${suffix}(${JSON.stringify(msg.args || {})})`
        : i18n('tool.none');
      log(`tool: ${msg.name || '—'}${suffix}`);
      if (msg.target_agent) _markAgentActive(msg.target_agent);
      break;
    }
    case 'response':
      // Display only — TTS audio arrives via the RTC track, not over WS.
      responseEl.textContent = msg.text;
      lastResponseText = msg.text;
      log(
        `response (llm=${msg.llm_ms}ms, history=${msg.history_turns ?? '?'}): ${msg.text}`
      );
      break;
    case 'push_tts': {
      // Server already plays the audio through the RTC track; this is
      // just a UI hint (and a chance to fire a desktop notification).
      const isMissed = msg.reason === 'missed_reminder';
      const prefix = isMissed ? i18n('push_tts.missed_reminder_prefix') : '';
      const fullText = prefix + msg.text;
      log(`push (${msg.reason}): ${fullText}`);
      if (document.hidden) _notify(fullText);
      break;
    }
    case 'replay':
      // No client-side cached buffer any more — server re-synthesises
      // and pushes through the RTC track.  Just log it.
      log(`replay: ${msg.mode}`);
      break;
    case 'response_end':
      // Image attaches are single-shot: server clears its slot at
      // pipeline start, so we mirror that on the UI by hiding the
      // preview the moment the turn wraps up.  Asking a follow-up
      // about the same image requires re-attaching it.
      if (attachPreviewEl && attachPreviewEl.style.display === 'block') {
        _hideAttachPreview();
      }
      log('response_end');
      break;
    case 'error':
      log(`ERROR: ${msg.message}`);
      break;
    case 'cancelled':
      log('cancelled');
      break;
    case 'ping':
      break;
    case 'history_reset':
      log('history cleared');
      break;
    case 'auth': {
      // Voice-side passphrase succeeded.  Cookie session is unaffected
      // — this only opens a 5-minute "tier-2 from voice" window — but
      // the UI may want to indicate it briefly and refresh the
      // pending list since approve actions will now succeed.
      log(`auth: voice window opened for profile=${msg.profile_id}`);
      if (typeof loadPending === 'function') loadPending();
      break;
    }
    case 'attach_image_ack': {
      if (msg.ok) {
        if (msg.cleared !== undefined) {
          log(`attach cleared (was ${msg.cleared ? 'present' : 'empty'})`);
        } else {
          log(`attached ${msg.name || 'image'} (${msg.bytes_b64} b64 chars)`);
        }
      } else {
        log(`attach error: ${msg.error || 'unknown'}`);
        // Server rejected — drop the local preview so user knows.
        if (typeof _hideAttachPreview === 'function') _hideAttachPreview();
      }
      break;
    }
    case 'voicemail_play': {
      // The orchestrator's inbox_read tool fired — play the original
      // recording locally.  The TTS intro ("Message from X:") is
      // already arriving through the RTC track; we just queue the
      // raw wav playback right after.  The hidden <audio> element
      // serves as the player so we don't fight the RTC track.
      const url = `/api/voicemail/${msg.message_id}/audio`;
      log(`voicemail_play: id=${msg.message_id} from=${msg.from_name || '?'}`);
      _playVoicemailAudio(url, msg.message_id);
      // Auto-refresh the inbox panel if it's open so the listened-flag
      // flips and the badge count updates.
      if (typeof loadInbox === 'function') loadInbox();
      break;
    }
    case 'voicemail_arrived': {
      // Pipeline pushes this when someone saves a voicemail for a
      // logged-in profile.  Three responses, in order of "did the user
      // ask for them": chime (gated by the mute-sounds checkbox so a
      // notif during a meeting stays silent), desktop notification
      // (only when the tab is hidden and the user pre-granted it),
      // and a live inbox refresh so the badge updates even before
      // the user opens the panel.
      log(`voicemail_arrived: id=${msg.message_id} from=${msg.from_name || '?'}`);
      _playChime();
      _maybeNotifyVoicemail(msg);
      if (typeof loadInbox === 'function') loadInbox();
      break;
    }
    default:
      log(`unknown msg: ${msg.type}`);
  }
}

// ── Voicemail-arrived notification helpers ──────────────────────────────
//
// Three layers, increasingly intrusive: chime → desktop notification →
// inbox badge refresh (handled inline).  Each one is gated so a user
// who doesn't want it can shut it off (mute-sounds checkbox, Notification
// permission denied).  We deliberately keep this off the WebAudio path
// used for state cues (wake/followup/session_end) — those are tied to
// the conversation FSM and louder; this is a passive ping.
let _chimeCtx = null;
function _playChime() {
  if (muteSoundsEl?.checked) return;
  try {
    if (!_chimeCtx) _chimeCtx = new AudioContext();
    const ctx = _chimeCtx;
    const t0 = ctx.currentTime;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.connect(g);
    g.connect(ctx.destination);
    osc.type = 'sine';
    // Short rising sweep, 0.15 s — distinct from the wake/followup
    // beep (which are flat tones) so a long-time user learns the
    // difference without thinking about it.
    osc.frequency.setValueAtTime(660, t0);
    osc.frequency.exponentialRampToValueAtTime(990, t0 + 0.15);
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(0.06, t0 + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.15);
    osc.start(t0);
    osc.stop(t0 + 0.2);
  } catch (e) {
    log(`chime failed: ${e.message}`);
  }
}

function _maybeNotifyVoicemail(msg) {
  // Only fire a desktop Notification when (a) the user already granted
  // it AND (b) the tab is hidden — otherwise the in-page chime + badge
  // is enough and a duplicate desktop popup would be annoying.
  if (typeof Notification === 'undefined') return;
  if (Notification.permission !== 'granted') return;
  if (!document.hidden) return;
  try {
    const from = msg.from_name || i18n('inbox.guest');
    const title = i18n('inbox.notify_title');
    const body = i18n('inbox.notify_body').replace('{from_name}', from);
    const n = new Notification(title, { body, tag: `voicemail-${msg.message_id}` });
    n.onclick = () => {
      try { window.focus(); n.close(); } catch (e) { /* ignore */ }
    };
  } catch (e) {
    log(`notify failed: ${e.message}`);
  }
}

// One-shot per session: if permission is still `default` (never asked),
// prompt once on successful login.  No nag — if denied, we never ask
// again from this tab.  On grant, we ALSO register the service worker
// and subscribe to Web Push so the user gets voicemail notifications
// even when every tab is closed.
let _askedPermissionThisSession = false;
function _requestNotificationPermission() {
  if (_askedPermissionThisSession) return;
  if (typeof Notification === 'undefined') return;
  if (Notification.permission !== 'default') return;
  _askedPermissionThisSession = true;
  try {
    const p = Notification.requestPermission();
    if (p && typeof p.then === 'function') {
      p.then((s) => {
        log(`notification permission: ${s}`);
        if (s === 'granted') _setupWebPush();
      });
    }
  } catch (e) {
    log(`notify perm request failed: ${e.message}`);
  }
}

// ── Web Push (VAPID) registration ────────────────────────────────────────
//
// Closed-browser delivery: even when every tab is closed, the push
// service can wake the Service Worker (sw.js) and surface a system
// notification.  This is purely additive over the WS `voicemail_arrived`
// event — when the tab IS open we get both, and the in-page chime +
// inbox refresh take precedence over the OS toast (the SW dedupes via
// `tag`).
//
// Triggered on first successful login + on every re-login.  Idempotent:
// pushManager.subscribe() returns the existing subscription if one is
// already live for this origin, and the server upserts by endpoint so
// duplicates can't accumulate.

// Convert the VAPID public key (urlsafe-base64 without padding) into
// the Uint8Array shape pushManager.subscribe expects.  Mirror of the
// helper in sw.js — keep them identical if you touch one.
function _b64UrlToUint8Array(b64Url) {
  const padding = '='.repeat((4 - (b64Url.length % 4)) % 4);
  const b64 = (b64Url + padding).replace(/-/g, '+').replace(/_/g, '/');
  const raw = atob(b64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

let _pushSetupInFlight = false;
async function _setupWebPush() {
  if (_pushSetupInFlight) return;
  if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
    log('webpush: not supported in this browser');
    return;
  }
  if (typeof Notification === 'undefined' || Notification.permission !== 'granted') {
    return;
  }
  _pushSetupInFlight = true;
  try {
    // Register at root scope so '/' (the SPA) is controlled.  Safe to
    // call repeatedly — the browser returns the existing registration
    // if one already matches the script URL.
    const reg = await navigator.serviceWorker.register('/sw.js');
    log(`webpush: SW ready (scope=${reg.scope})`);

    // Pull the VAPID public key fresh on every login — cheap, and
    // means a key rotation on the server picks up next sign-in.
    const r = await fetch('/api/push/vapid_public_key');
    if (!r.ok) {
      log(`webpush: vapid key fetch failed (HTTP ${r.status})`);
      return;
    }
    const { public_key } = await r.json();
    const applicationServerKey = _b64UrlToUint8Array(public_key);

    let sub = await reg.pushManager.getSubscription();
    if (!sub) {
      sub = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey,
      });
      log('webpush: new push subscription created');
    } else {
      log('webpush: reusing existing push subscription');
    }
    // Hand the subscription to the orchestrator so it can fan out
    // voicemail pushes when the tab is closed.  Upserts on the server
    // side — calling this on every login is fine.
    const post = await fetch('/api/push/subscribe', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sub.toJSON()),
    });
    if (!post.ok) {
      log(`webpush: server-side register failed (HTTP ${post.status})`);
    } else {
      log('webpush: subscription registered with server');
    }
  } catch (e) {
    log(`webpush: setup failed: ${e.message}`);
  } finally {
    _pushSetupInFlight = false;
  }
}

// On logout — tell the server to drop our subscription so we stop
// receiving notifications for the now-anonymous browser.  We keep the
// browser-side PushSubscription alive (calling .unsubscribe() can
// surface a permission prompt next login) — the server side dropping
// the row is enough to stop delivery.
async function _teardownWebPush() {
  try {
    if (!('serviceWorker' in navigator)) return;
    const reg = await navigator.serviceWorker.getRegistration('/');
    if (!reg) return;
    const sub = await reg.pushManager.getSubscription();
    if (!sub) return;
    await fetch('/api/push/subscribe', {
      method: 'DELETE',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ endpoint: sub.endpoint }),
    });
    log('webpush: server-side subscription removed on logout');
  } catch (e) {
    log(`webpush: teardown failed: ${e.message}`);
  }
}

// Service-worker → page channel.  Used when notificationclick focuses
// an existing tab and wants main.js to scroll the inbox panel to the
// linked voicemail.  Soft-fails if the panel isn't mounted (e.g. user
// is anonymous in another tab) — best effort.
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.addEventListener('message', (event) => {
    const data = event.data || {};
    if (data.type !== 'sw_notification_click') return;
    log(`webpush: SW notif click → voicemail=${data.voicemail_id ?? '?'}`);
    if (typeof loadInbox === 'function') {
      // Refresh the inbox so the user's panel reflects the new row;
      // the panel-open user gesture is implicit from the OS notif click.
      loadInbox();
    }
  });
}

// Single-instance audio player for voicemail playback.  Created lazily
// the first time the server signals voicemail_play.  Reused so a fast
// follow-up play stops the previous one cleanly.
let _voicemailPlayer = null;
function _playVoicemailAudio(url, messageId) {
  if (!_voicemailPlayer) {
    _voicemailPlayer = document.createElement('audio');
    _voicemailPlayer.style.display = 'none';
    _voicemailPlayer.preload = 'none';
    document.body.appendChild(_voicemailPlayer);
    _voicemailPlayer.addEventListener('ended', async () => {
      // Mark listened on the server so the badge updates next refresh.
      // Best-effort — a failure here doesn't surface to the user.
      const id = _voicemailPlayer.dataset.messageId;
      if (!id) return;
      try {
        await fetch(`/api/voicemail/${id}/listened`, {
          method: 'POST',
          credentials: 'same-origin',
        });
      } catch (e) { /* ignore */ }
    });
  }
  _voicemailPlayer.src = url;
  _voicemailPlayer.dataset.messageId = String(messageId);
  // Best-effort play; browsers may block if no user gesture happened
  // recently, but the WS event itself was triggered by one (Talk/PTT).
  _voicemailPlayer.play().catch((e) => log(`voicemail play blocked: ${e.message}`));
}

function cleanup() {
  if (pc) {
    try { pc.close(); } catch {}
    pc = null;
  }
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (micSource) { micSource.disconnect(); micSource = null; }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (ws) {
    try { ws.close(); } catch {}
    ws = null;
  }
  if (remoteTtsEl) remoteTtsEl.srcObject = null;
  wakeDetector = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  cancelBtn.disabled = true;
  pttBtn.disabled = true;
  setState('—');
  sampleRateEl.textContent = '—';
  if (wakeScoreEl) wakeScoreEl.textContent = '—';
}

function cancel() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'cancel' }));
  }
}

// ── Image attach (Sprint 6) ──────────────────────────────────────────────
//
// The server keeps at most one pending image per WS session.  Picking
// or dropping a new image replaces whatever was queued.  The image is
// sent as a base64 data URL right away (so the user sees the preview
// even before they speak), and the server stages it until the next
// pipeline run — which short-circuits the agent loop and routes
// transcript + image to Claude vision.

function _showAttachPreview(name, dataUrl, bytes) {
  if (!attachPreviewEl) return;
  if (attachThumbEl) attachThumbEl.src = dataUrl;
  if (attachMetaEl) {
    const kb = (bytes / 1024).toFixed(1);
    attachMetaEl.textContent = i18n('attach.meta', { name, kb });
  }
  attachPreviewEl.style.display = 'block';
}

function _hideAttachPreview() {
  if (!attachPreviewEl) return;
  attachPreviewEl.style.display = 'none';
  if (attachThumbEl) attachThumbEl.removeAttribute('src');
  if (attachMetaEl) attachMetaEl.textContent = '';
  if (attachInputEl) attachInputEl.value = '';
}

function _readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

async function handleAttachFile(file) {
  if (!file) return;
  if (!file.type || !file.type.startsWith('image/')) {
    log(`attach: skipping non-image ${file.name} (${file.type})`);
    return;
  }
  // Soft cap on raw bytes — Claude vision tops out around 5 MB binary;
  // larger inputs cost more and rarely improve quality.  Hard cap
  // happens server-side; this is just an early UX warning.
  const HARD_LIMIT = 8 * 1024 * 1024;
  if (file.size > HARD_LIMIT) {
    log(`attach: ${file.name} too large (${(file.size/1024/1024).toFixed(1)} MB > ${HARD_LIMIT/1024/1024} MB)`);
    return;
  }
  let dataUrl;
  try {
    dataUrl = await _readFileAsDataURL(file);
  } catch (e) {
    log(`attach: file read error: ${e?.message || e}`);
    return;
  }
  // Show preview immediately; ack-or-error from server flips the meta
  // line if anything went wrong.
  _showAttachPreview(file.name, dataUrl, file.size);
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    log('attach: WS not connected — preview shown locally only');
    return;
  }
  ws.send(
    JSON.stringify({
      type: 'attach_image',
      name: file.name,
      data: dataUrl,
    })
  );
}

function detachImage() {
  _hideAttachPreview();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'detach_image' }));
  }
}

if (attachBtnEl && attachInputEl) {
  attachBtnEl.addEventListener('click', () => attachInputEl.click());
  attachInputEl.addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (f) handleAttachFile(f);
  });
}
if (attachClearEl) {
  attachClearEl.addEventListener('click', detachImage);
}

// Drag-drop anywhere on the document — capture image, swallow event.
// We intentionally use document-level listeners (not just the attach
// area) so the user can drop on the page without aiming.  Drop on
// inputs/text-areas still works for native pastes because we don't
// preventDefault on those.
if (window && document) {
  ['dragenter', 'dragover'].forEach((ev) =>
    document.addEventListener(ev, (e) => {
      if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files')) {
        e.preventDefault();
        e.stopPropagation();
      }
    })
  );
  document.addEventListener('drop', (e) => {
    if (!e.dataTransfer) return;
    const file = e.dataTransfer.files?.[0];
    if (file && file.type && file.type.startsWith('image/')) {
      e.preventDefault();
      e.stopPropagation();
      handleAttachFile(file);
    }
  });
  // Clipboard-paste support — Cmd-V / Ctrl-V with an image on the
  // clipboard (screenshot tools love this path).  No preventDefault
  // unless the clipboard actually carries an image.
  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items || [];
    for (const it of items) {
      if (it.kind === 'file' && it.type?.startsWith('image/')) {
        const file = it.getAsFile();
        if (file) {
          e.preventDefault();
          handleAttachFile(file);
          break;
        }
      }
    }
  });
}

let continuationCountdownTimer = null;
let continuationDeadline = 0;

function clearContinuationCountdown() {
  if (continuationCountdownTimer) {
    clearInterval(continuationCountdownTimer);
    continuationCountdownTimer = null;
  }
  const el = document.querySelector('#countdown');
  if (el) el.textContent = '';
}

function startContinuationCountdown(timeoutS) {
  clearContinuationCountdown();
  continuationDeadline = Date.now() + timeoutS * 1000;
  const el = document.querySelector('#countdown');
  if (!el) return;
  const tick = () => {
    const remaining = Math.max(0, Math.ceil((continuationDeadline - Date.now()) / 1000));
    el.textContent = i18n('state.continuation.countdown', { n: remaining });
    if (remaining === 0) clearContinuationCountdown();
  };
  tick();
  continuationCountdownTimer = setInterval(tick, 1000);
}

let pttActive = false;
function pttStart() {
  if (pttActive || !ws || ws.readyState !== WebSocket.OPEN) return;
  pttActive = true;
  pttBtn.classList.add('pressed');
  ws.send(JSON.stringify({ type: 'ptt_start' }));
}
function pttEnd() {
  if (!pttActive || !ws || ws.readyState !== WebSocket.OPEN) return;
  pttActive = false;
  pttBtn.classList.remove('pressed');
  ws.send(JSON.stringify({ type: 'ptt_end' }));
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', cleanup);
cancelBtn.addEventListener('click', cancel);

pttBtn.addEventListener('mousedown', pttStart);
pttBtn.addEventListener('mouseup', pttEnd);
pttBtn.addEventListener('mouseleave', pttEnd);
pttBtn.addEventListener('touchstart', (e) => { e.preventDefault(); pttStart(); });
pttBtn.addEventListener('touchend',   (e) => { e.preventDefault(); pttEnd(); });

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat && document.activeElement?.tagName !== 'INPUT') {
    e.preventDefault();
    pttStart();
  }
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space') {
    e.preventDefault();
    pttEnd();
  }
});

// ---------------------------------------------------------------------------
// Speaker enrollment (uses a separate getUserMedia stream, captures PCM via
// the same worklet, POSTs to /api/speakers/enroll).  The profiles list also
// owns the per-speaker TTS voice picker: each enrolled person can choose
// which XTTS voice they want answers in (defaults to the server-wide voice
// when blank/auto).
// ---------------------------------------------------------------------------

// Catalogue of TTS voices, fetched from /api/voices.  Two flavours:
//   • builtIn — fixed XTTS speaker bank, names like "Claribel Dervla"
//   • custom  — user-recorded reference WAVs, referenced by "clone:<id>"
// The built-in list stays stable for the life of the page (XTTS speaker
// bank is fixed at model load time), but the custom list changes when
// the user records or deletes a clone, so we invalidate it on those
// flows via _invalidateVoiceCatalogue().
let _voiceCatalogue = null;

function _invalidateVoiceCatalogue() {
  _voiceCatalogue = null;
}

async function loadVoiceCatalogue() {
  if (_voiceCatalogue !== null) return _voiceCatalogue;
  try {
    const r = await fetch('/api/voices');
    const data = await r.json();
    _voiceCatalogue = {
      builtIn: data.voices ?? [],
      custom: data.custom_voices ?? [],
    };
  } catch (e) {
    log(`failed to load voice list: ${e.message}`);
    _voiceCatalogue = { builtIn: [], custom: [] };
  }
  return _voiceCatalogue;
}

async function loadSpeakers() {
  try {
    // Voices in parallel so the first paint already has the dropdown
    // options populated — no flash of empty <select>.
    const [r, voices] = await Promise.all([
      fetch(`/api/speakers?client_id=${encodeURIComponent(clientId)}`),
      loadVoiceCatalogue(),
    ]);
    const data = await r.json();
    renderSpeakers(data.speakers ?? [], voices);
  } catch (e) {
    log(`error loading profiles: ${e.message}`);
  }
}

function _escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderSpeakers(speakers, voices) {
  const list = document.querySelector('#speaker-list');
  if (!list) return;
  list.innerHTML = '';
  if (!speakers.length) {
    list.innerHTML = `<li><small style="color:var(--pico-muted-color)">${i18n('speaker.no_profiles')}</small></li>`;
    return;
  }
  const builtIn = voices?.builtIn ?? [];
  const custom = voices?.custom ?? [];
  for (const sp of speakers) {
    const li = document.createElement('li');
    li.style.cssText = 'display:grid;grid-template-columns:1fr auto auto;align-items:center;gap:0.5rem;margin-bottom:0.35rem';
    const samples = sp.samples ?? 1;
    const badge = samples > 1
      ? ` <small style="color:var(--pico-muted-color)">×${samples}</small>`
      : '';
    // Build the voice <select>.  Three sections:
    //   • default              — empty value, falls back to server-wide voice
    //   • <optgroup Custom>   — user-recorded clones, value = "clone:<id>"
    //   • <optgroup Built-in> — XTTS built-ins, value = speaker name
    // The custom group is omitted entirely when empty so the dropdown
    // doesn't show an awkward heading with no rows.
    const optsParts = [`<option value="">${_escapeHtml(i18n('speaker.voice_default'))}</option>`];
    if (custom.length) {
      const customOpts = custom.map((v) => {
        const val = `clone:${v.id}`;
        const sel = val === sp.voice ? ' selected' : '';
        return `<option value="${val}"${sel}>${_escapeHtml(v.name)}</option>`;
      }).join('');
      optsParts.push(`<optgroup label="${_escapeHtml(i18n('speaker.optgroup_custom'))}">${customOpts}</optgroup>`);
    }
    if (builtIn.length) {
      const builtInOpts = builtIn.map((v) => {
        const sel = v === sp.voice ? ' selected' : '';
        return `<option value="${_escapeHtml(v)}"${sel}>${_escapeHtml(v)}</option>`;
      }).join('');
      optsParts.push(`<optgroup label="${_escapeHtml(i18n('speaker.optgroup_builtin'))}">${builtInOpts}</optgroup>`);
    }
    li.innerHTML = `
      <span>${_escapeHtml(sp.name)}${badge}</span>
      <select data-id="${sp.id}"
        style="padding:0.15rem 0.35rem;font-size:0.75rem;margin:0;min-width:9rem">
        ${optsParts.join('')}
      </select>
      <button class="outline secondary" data-id="${sp.id}"
        style="padding:0.15rem 0.5rem;font-size:0.75rem;margin:0">✕</button>`;
    li.querySelector('select').addEventListener('change', (e) =>
      patchSpeakerVoice(sp.id, sp.name, e.target.value),
    );
    li.querySelector('button').addEventListener('click', () => deleteSpeaker(sp.id, sp.name));
    list.appendChild(li);
  }
}

async function patchSpeakerVoice(id, name, voice) {
  try {
    const r = await fetch(`/api/speakers/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tts_voice: voice || null }),
    });
    const data = await r.json();
    if (data.ok) {
      log(`speaker: "${name}" → voice ${voice || 'default'}`);
    } else {
      log(`speaker: failed to update voice (${data.error ?? 'unknown'})`);
    }
  } catch (e) {
    log(`speaker: voice update error: ${e.message}`);
  }
}

async function deleteSpeaker(id, name) {
  await fetch(`/api/speakers/${id}`, { method: 'DELETE' });
  log(`speaker: deleted profile "${name}"`);
  loadSpeakers();
}

let _enrollActive = false;

async function startEnrollFlow() {
  const nameInput = document.querySelector('#enroll-name');
  const enrollBtn = document.querySelector('#enroll-btn');
  const status = document.querySelector('#enroll-status');
  const name = nameInput?.value.trim();

  if (!name) { log('enter a name before recording'); return; }
  if (_enrollActive) return;
  _enrollActive = true;
  if (enrollBtn) enrollBtn.disabled = true;

  let stream = null;
  let actx = null;
  let node = null;
  let src = null;
  const chunks = [];

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: 16000,
      },
    });
    actx = new AudioContext({ sampleRate: 16000 });
    await actx.audioWorklet.addModule('./worklet.js');
    src = actx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(actx, 'pcm-recorder');
    node.port.onmessage = (e) => chunks.push(new Int16Array(e.data));
    src.connect(node);

    for (let i = 5; i > 0; i--) {
      if (status) status.textContent = i18n('speaker.recording', { s: i });
      await new Promise((r) => setTimeout(r, 1000));
    }
  } finally {
    if (node) { node.disconnect(); }
    if (src) { src.disconnect(); }
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (actx) await actx.close().catch(() => {});
  }

  if (status) status.textContent = i18n('speaker.uploading');

  const total = chunks.reduce((s, c) => s + c.length, 0);
  const pcm = new Int16Array(total);
  let off = 0;
  for (const c of chunks) { pcm.set(c, off); off += c.length; }

  try {
    const form = new FormData();
    form.append(
      'audio',
      new Blob([pcm.buffer], { type: 'application/octet-stream' }),
      'voice.pcm',
    );
    const url =
      `/api/speakers/enroll?client_id=${encodeURIComponent(clientId)}&name=${encodeURIComponent(name)}`;
    const r = await fetch(url, { method: 'POST', body: form });
    const data = await r.json();
    if (data.ok) {
      if (status) status.textContent = i18n('speaker.saved', { name });
      log(`speaker: enrolled "${name}" (id=${data.id})`);
      if (nameInput) nameInput.value = '';
      loadSpeakers();
    } else {
      if (status) status.textContent = i18n('speaker.error', { err: data.error });
      log(`speaker enroll error: ${data.error}`);
    }
  } catch (e) {
    if (status) status.textContent = `❌ ${e.message}`;
    log(`speaker enroll failed: ${e.message}`);
  }

  _enrollActive = false;
  if (enrollBtn) enrollBtn.disabled = false;
}

document.querySelector('#enroll-btn')?.addEventListener('click', startEnrollFlow);

// ---------------------------------------------------------------------------
// Custom output voices — record a 6 s reference clip that xtts-server uses
// for on-the-fly voice cloning.  Independent of speaker enrollment (which
// identifies who's TALKING) — this section is about who the assistant
// SOUNDS like.  Each saved row appears as a "Custom" option in every
// speaker's voice dropdown (value = "clone:<id>"); selecting it pins that
// voice for the speaker via the same PATCH /api/speakers/{id} endpoint.
// ---------------------------------------------------------------------------

const CLONE_RECORD_SECONDS = 6;

async function loadCustomVoices() {
  try {
    const r = await fetch('/api/custom_voices');
    const data = await r.json();
    renderCustomVoices(data.custom_voices ?? []);
  } catch (e) {
    log(`error loading custom voices: ${e.message}`);
  }
}

function renderCustomVoices(voices) {
  const list = document.querySelector('#custom-voice-list');
  if (!list) return;
  list.innerHTML = '';
  if (!voices.length) {
    list.innerHTML = `<li><small style="color:var(--pico-muted-color)">${i18n('voice.no_clones')}</small></li>`;
    return;
  }
  for (const v of voices) {
    const li = document.createElement('li');
    li.style.cssText = 'display:grid;grid-template-columns:1fr auto;align-items:center;gap:0.5rem;margin-bottom:0.35rem';
    li.innerHTML = `
      <span>${_escapeHtml(v.name)} <small style="color:var(--pico-muted-color)">#${v.id}</small></span>
      <button class="outline secondary" data-id="${v.id}"
        style="padding:0.15rem 0.5rem;font-size:0.75rem;margin:0">✕</button>`;
    li.querySelector('button').addEventListener('click', () => deleteCustomVoice(v.id, v.name));
    list.appendChild(li);
  }
}

let _cloneActive = false;

async function startCloneRecording() {
  const nameInput = document.querySelector('#clone-name');
  const btn = document.querySelector('#clone-btn');
  const status = document.querySelector('#clone-status');
  const name = nameInput?.value.trim();

  if (!name) { log('enter a name before recording'); return; }
  if (_cloneActive) return;
  _cloneActive = true;
  if (btn) btn.disabled = true;

  let stream = null;
  let actx = null;
  let node = null;
  let src = null;
  const chunks = [];

  try {
    // Same capture chain as the enrollment flow — mono int16 16 kHz PCM
    // via the existing pcm-recorder worklet.  XTTS-v2's conditioning
    // encoder is sample-rate agnostic for the reference audio; we wrap
    // the raw PCM in a WAV header server-side.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: 16000,
      },
    });
    actx = new AudioContext({ sampleRate: 16000 });
    await actx.audioWorklet.addModule('./worklet.js');
    src = actx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(actx, 'pcm-recorder');
    node.port.onmessage = (e) => chunks.push(new Int16Array(e.data));
    src.connect(node);

    for (let i = CLONE_RECORD_SECONDS; i > 0; i--) {
      if (status) status.textContent = i18n('voice.recording', { s: i });
      await new Promise((r) => setTimeout(r, 1000));
    }
  } finally {
    if (node) { node.disconnect(); }
    if (src) { src.disconnect(); }
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (actx) await actx.close().catch(() => {});
  }

  if (status) status.textContent = i18n('voice.uploading');

  const total = chunks.reduce((s, c) => s + c.length, 0);
  const pcm = new Int16Array(total);
  let off = 0;
  for (const c of chunks) { pcm.set(c, off); off += c.length; }

  try {
    const form = new FormData();
    form.append(
      'audio',
      new Blob([pcm.buffer], { type: 'application/octet-stream' }),
      'voice.pcm',
    );
    const url = `/api/custom_voices/record?name=${encodeURIComponent(name)}`;
    const r = await fetch(url, { method: 'POST', body: form });
    const data = await r.json();
    if (data.ok) {
      if (status) status.textContent = i18n('voice.saved', { name });
      log(`custom voice: saved "${name}" (id=${data.id})`);
      if (nameInput) nameInput.value = '';
      _invalidateVoiceCatalogue();
      // Refresh both the custom-voice list and the speakers section
      // (so the new clone appears in every speaker's <select>).
      await loadCustomVoices();
      await loadSpeakers();
    } else {
      if (status) status.textContent = i18n('voice.error', { err: data.error });
      log(`custom voice error: ${data.error}`);
    }
  } catch (e) {
    if (status) status.textContent = `❌ ${e.message}`;
    log(`custom voice failed: ${e.message}`);
  }

  _cloneActive = false;
  if (btn) btn.disabled = false;
}

async function deleteCustomVoice(id, name) {
  await fetch(`/api/custom_voices/${id}`, { method: 'DELETE' });
  log(`custom voice: deleted «${name}»`);
  _invalidateVoiceCatalogue();
  await loadCustomVoices();
  await loadSpeakers();
}

document.querySelector('#clone-btn')?.addEventListener('click', startCloneRecording);

loadSpeakers();
loadCustomVoices();

// ---------------------------------------------------------------------------
// "Statistics" dashboard — token usage and projected cost.
//
// Polls GET /api/stats?range=… on demand: at page load (so the section
// has data the moment the user expands it), when the period <select>
// changes, and when the refresh button is clicked.
//
// Chart.js is loaded from CDN in index.html.  If it never loaded (offline
// dev, blocked CDN) we degrade gracefully: the section shows the cost
// summary and a friendly note instead of an empty chart frame.
// ---------------------------------------------------------------------------

const statsRangeEl   = document.querySelector('#stats-range');
const statsRefreshEl = document.querySelector('#stats-refresh');
const statsStatusEl  = document.querySelector('#stats-status');
const statsCostEl    = document.querySelector('#stats-cost');
const statsSectionEl = document.querySelector('#stats-section');

// Hold onto Chart instances so we can destroy them before re-render —
// otherwise Chart.js stacks them on top of each other on every refresh
// and the canvas grows infinitely.
const _statsCharts = { daily: null, perTool: null, perUser: null };

function _hasChartJs() {
  return typeof window !== 'undefined' && typeof window.Chart === 'function';
}

function _fmtTokens(n) {
  if (!n) return '0';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'k';
  return String(n);
}

function _fmtUsd(n) {
  if (!Number.isFinite(n)) return '$0.00';
  // Sub-cent precision for tiny dev workloads, normal 2dp once we cross $1.
  if (n < 0.01) return `$${n.toFixed(5)}`;
  if (n < 1)    return `$${n.toFixed(4)}`;
  return `$${n.toFixed(2)}`;
}

async function loadStats(range) {
  if (!statsStatusEl) return;
  statsStatusEl.textContent = i18n('stats.loading');
  try {
    const r = await fetch(`/api/stats?range=${encodeURIComponent(range)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderStats(data);
    statsStatusEl.textContent = i18n('stats.updated', { time: new Date().toLocaleTimeString() });
  } catch (e) {
    statsStatusEl.textContent = i18n('stats.error', { err: e.message });
    log(`stats: ${e.message}`);
  }
}

function renderStats(data) {
  renderCostSummary(data);
  renderDailyChart(data.daily || []);
  renderPerToolChart(data.per_tool || []);
  renderPerUserChart(data.per_user || []);
}

function renderCostSummary(data) {
  if (!statsCostEl) return;
  const cost = data.cost || {};
  const claude = cost['claude-sonnet-4.5'];
  const mini   = cost['gpt-4o-mini'];
  const totalPrompt     = claude ? claude.prompt_tokens     : 0;
  const totalCompletion = claude ? claude.completion_tokens : 0;
  const totalTokens = totalPrompt + totalCompletion;
  if (!totalTokens) {
    statsCostEl.innerHTML = `<span>${i18n('stats.no_data')}</span>`;
    return;
  }
  const claudeUsd = claude ? claude.cost_usd : 0;
  const miniUsd   = mini   ? mini.cost_usd   : 0;
  // Build the two lines via t() so they update when the language changes.
  // The token counts are pre-formatted strings, not raw numbers.
  const line1 = i18n('stats.tokens_summary', {
    total:      _fmtTokens(totalTokens),
    prompt:     _fmtTokens(totalPrompt),
    completion: _fmtTokens(totalCompletion),
  });
  const line2 = i18n('stats.cost_if', {
    usd_claude: `<strong>${_fmtUsd(claudeUsd)}</strong>`,
    usd_mini:   `<strong>${_fmtUsd(miniUsd)}</strong>`,
  });
  statsCostEl.innerHTML = `<div>${line1}</div><div>${line2}</div>`;
}

function _destroyChart(key) {
  const c = _statsCharts[key];
  if (c) {
    try { c.destroy(); } catch {}
    _statsCharts[key] = null;
  }
}

function renderDailyChart(daily) {
  const canvas = document.querySelector('#daily-chart');
  if (!canvas) return;
  _destroyChart('daily');
  if (!_hasChartJs()) {
    // The Chart.js fallback message lives in <small> next to the chart.
    // Walk OUT to the .stats-card to find it; the canvas's *immediate*
    // parent is the fixed-height .chart-frame which holds no text.
    const card = canvas.closest('.stats-card');
    const cap = card?.querySelector('small');
    if (cap) cap.textContent = i18n('stats.chartjs_unavailable');
    return;
  }
  // Cap to the most recent 30 days even if the API returns more.  At
  // 30+ daily bars the legend labels start overlapping and the chart
  // turns into mud; we keep the API generous in case the user wants
  // raw data via /api/stats but never render past this many.
  const tail = daily.slice(-30);
  const labels = tail.map((d) => d.day);
  const prompt = tail.map((d) => d.prompt_tokens);
  const completion = tail.map((d) => d.completion_tokens);
  _statsCharts.daily = new window.Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'prompt',     data: prompt,     backgroundColor: 'rgba(59,130,246,0.7)'  },
        { label: 'completion', data: completion, backgroundColor: 'rgba(16,185,129,0.7)' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { stacked: true },
        y: { stacked: true, beginAtZero: true },
      },
    },
  });
}

// Hard cap on how many rows we render in the horizontal-bar charts.
// Mirrors the server-side _DASHBOARD_TOP_N — having both is a belt-
// and-braces guard so a back-end that hasn't been redeployed can't
// nuke the UI with a 30-bar chart.  The bar charts overflow visually
// past ~10 rows on a 220-px frame.
const _STATS_TOP_N = 8;

function _truncateLabel(s, max = 22) {
  if (!s) return '(?)';
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

function renderPerToolChart(rows) {
  const canvas = document.querySelector('#per-tool-chart');
  if (!canvas) return;
  _destroyChart('perTool');
  if (!_hasChartJs()) return;
  const trimmed = rows.slice(0, _STATS_TOP_N);
  const labels = trimmed.map((r) => _truncateLabel(r.tool_name || '(unknown)'));
  const totals = trimmed.map((r) => (r.prompt_tokens || 0) + (r.completion_tokens || 0));
  _statsCharts.perTool = new window.Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'tokens (prompt + completion)',
          data: totals,
          backgroundColor: 'rgba(245,158,11,0.7)',
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } },
    },
  });
}

function renderPerUserChart(rows) {
  const canvas = document.querySelector('#per-user-chart');
  if (!canvas) return;
  _destroyChart('perUser');
  if (!_hasChartJs()) return;
  // Show first 8 chars of UUIDs so the bar labels stay readable;
  // anonymous traffic shows up as the literal "(anon)" bucket; the
  // server may return a long-tail bucket starting with a localised "more
  // clients (N)" phrase — normalise it to the i18n key so it renders in
  // the right lang regardless of the server's locale.
  const trimmed = rows.slice(0, _STATS_TOP_N);
  const labels = trimmed.map((r) => {
    if (r.client_id === '(anon)') return '(anon)';
    // Server may return a "more clients (N)" bucket in any locale; detect
    // by numeric tail and re-render via i18n.
    if (r.client_id) {
      const moreMatch = r.client_id.match(/(\d+)\s*\)?\s*$/);
      // The "more clients" bucket is not a UUID — it contains spaces and
      // letters, never just hex + dashes.  Accept any non-UUID-looking
      // client_id that ends with a number.
      if (moreMatch && !/^[0-9a-f-]{8,}$/i.test(r.client_id)) {
        return i18n('stats.more_clients', { n: moreMatch[1] });
      }
    }
    return String(r.client_id || '').slice(0, 8);
  });
  const totals = trimmed.map((r) => (r.prompt_tokens || 0) + (r.completion_tokens || 0));
  _statsCharts.perUser = new window.Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'tokens (prompt + completion)',
          data: totals,
          backgroundColor: 'rgba(139,92,246,0.7)',
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } },
    },
  });
}

function _currentRange() {
  return statsRangeEl?.value || 'week';
}

statsRangeEl?.addEventListener('change', () => loadStats(_currentRange()));
statsRefreshEl?.addEventListener('click', () => loadStats(_currentRange()));
// Also refresh whenever the user expands the <details>, so a stale view
// from a long-idle tab gets a refresh on demand.
statsSectionEl?.addEventListener('toggle', () => {
  if (statsSectionEl.open) loadStats(_currentRange());
});

// Initial paint only if the user already had the section expanded
// (browser persists <details open> state across reloads).  Otherwise
// wait until they actually open it — saves a load and an idle Chart.js
// instantiation on every page load.  See the 'toggle' listener above
// for the on-open fetch.
if (statsSectionEl?.open) loadStats(_currentRange());

// ---------------------------------------------------------------------------
// Sprint 2 — Phase D: auth banner, pending queue, memory editor, settings.
//
// All HTTP calls below use ``credentials: 'same-origin'`` so the
// ``va_session`` cookie set by /api/auth/login is sent automatically.
// The banner reads /api/me on load to decide which sub-block to show:
//
//   • 200 → "anonymous" hidden, "known" visible, tier-2 sections shown
//   • 401 + at least one profile has a passphrase set → "anonymous"
//     (login form)
//   • 401 + no profile has a passphrase yet → "needs-setup" (first-run
//     bootstrap)
//
// Tier-2 panels (Pending, Memory, Settings) live behind class
// `.auth-required`; their `style.display` is toggled wholesale.  Showing
// them while anonymous would just produce 401s on every interaction,
// which would be noisy and confusing.
// ---------------------------------------------------------------------------

// ── DOM refs ────────────────────────────────────────────────────────────

const authAnonymousEl     = document.querySelector('#auth-anonymous');
const authKnownEl         = document.querySelector('#auth-known');
const authNeedsSetupEl    = document.querySelector('#auth-needs-setup');
const authProfileSelect   = document.querySelector('#auth-profile');
const authPassphraseInput = document.querySelector('#auth-passphrase');
const authLoginBtn        = document.querySelector('#auth-login-btn');
const authStatusEl        = document.querySelector('#auth-status');
const authCurrentNameEl   = document.querySelector('#auth-current-name');
const authCurrentExpEl    = document.querySelector('#auth-current-expires');
const authLogoutBtn       = document.querySelector('#auth-logout-btn');
const authSetPassBtn      = document.querySelector('#auth-set-passphrase-btn');
const setupProfileSelect  = document.querySelector('#setup-profile');
const setupPassphraseEl   = document.querySelector('#setup-passphrase');
const setupBtn            = document.querySelector('#setup-btn');
const setupStatusEl       = document.querySelector('#setup-status');

const pendingPanelEl   = document.querySelector('#pending-panel');
const pendingBadgeEl   = document.querySelector('#pending-badge');
const pendingStatusEl  = document.querySelector('#pending-status');
const pendingListEl    = document.querySelector('#pending-list');
const pendingRefreshEl = document.querySelector('#pending-refresh');

const memoryPanelEl  = document.querySelector('#memory-panel');
const memoryEditorEl = document.querySelector('#memory-editor');
const memorySaveBtn  = document.querySelector('#memory-save');
const memoryReloadEl = document.querySelector('#memory-reload');
const memoryStatusEl = document.querySelector('#memory-status');

const settingsPanelEl    = document.querySelector('#settings-panel');
const settingsLangEl     = document.querySelector('#settings-language');
const settingsFormEl     = document.querySelector('#settings-formality');
const settingsVoiceEl    = document.querySelector('#settings-tts-voice');
const settingsStyleEl    = document.querySelector('#settings-style-prompt');
const settingsRawEl      = document.querySelector('#settings-raw');
const settingsSaveBtn    = document.querySelector('#settings-save');
const settingsReloadBtn  = document.querySelector('#settings-reload');
const settingsStatusEl   = document.querySelector('#settings-status');

// ── Module-level state ──────────────────────────────────────────────────
// `_currentProfile` is null when anonymous, otherwise the {id,name} of
// the logged-in profile.  Used by every tier-2 fetch to construct
// /api/users/<id>/<...> URLs.
let _currentProfile = null;

// ── Helpers ─────────────────────────────────────────────────────────────

function _setAuthBlock(name) {
  // Show exactly one of the three blocks inside #auth-banner.  Names:
  // 'anonymous' | 'known' | 'needs-setup' | 'loading' (all hidden).
  authAnonymousEl.style.display  = name === 'anonymous'   ? '' : 'none';
  authKnownEl.style.display      = name === 'known'       ? '' : 'none';
  authNeedsSetupEl.style.display = name === 'needs-setup' ? '' : 'none';
}

function _setTierTwoVisible(visible) {
  // Panels behind `.auth-required` are show/hidden together.  Use
  // `display:none` rather than `hidden` so <details> doesn't try to
  // animate its collapse box during the transition.
  document.querySelectorAll('.auth-required').forEach((el) => {
    el.style.display = visible ? '' : 'none';
  });
}

function _fmtExpiry(epoch) {
  if (!epoch) return '—';
  const d = new Date(epoch * 1000);
  return d.toLocaleString();
}

async function _fetchProfilesForLogin() {
  // Reuse /api/speakers — it's per-client_id, which matches the
  // login-from-this-browser model.  Returns [{id, name, ...}, ...].
  try {
    const r = await fetch(`/api/speakers?client_id=${encodeURIComponent(clientId)}`);
    const data = await r.json();
    return data.speakers || [];
  } catch (e) {
    log(`auth: cannot list profiles: ${e.message}`);
    return [];
  }
}

function _populateProfileSelect(selectEl, profiles) {
  // Wipe existing <option>s except the placeholder at index 0.
  while (selectEl.children.length > 1) selectEl.removeChild(selectEl.lastChild);
  for (const p of profiles) {
    const opt = document.createElement('option');
    opt.value = String(p.id);
    opt.textContent = p.name;
    selectEl.appendChild(opt);
  }
}

// ── Main coordinator ────────────────────────────────────────────────────

async function loadAuthState() {
  let me = null;
  try {
    const r = await fetch('/api/me', { credentials: 'same-origin' });
    if (r.ok) me = await r.json();
  } catch (e) {
    // Network error — treat as anonymous; we'll degrade the banner.
    log(`auth: /api/me failed: ${e.message}`);
  }

  const profiles = await _fetchProfilesForLogin();

  if (me) {
    // Resolve the profile name from the locally-known list (best-effort
    // — if the profile is on another client, we just show its id).
    const match = profiles.find((p) => p.id === me.profile_id);
    _currentProfile = { id: me.profile_id, name: match?.name || i18n('auth.profile_number', { id: me.profile_id }) };
    authCurrentNameEl.textContent = _currentProfile.name;
    authCurrentExpEl.textContent  = _fmtExpiry(me.expires_at);
    _setAuthBlock('known');
    _setTierTwoVisible(true);
    // First successful login of this tab session: opportunistically
    // ask for Notification permission.  Default-deny browsers (Safari)
    // just don't show a prompt; granted ones light up the desktop
    // toast on `voicemail_arrived` when the tab is hidden.  When
    // permission is ALREADY granted (e.g. user logged in once before),
    // jump straight to subscribing — _requestNotificationPermission
    // would no-op on permission != 'default'.
    if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      _setupWebPush();
    } else {
      _requestNotificationPermission();
    }
    // Refresh tier-2 sections opportunistically; the user will likely
    // open one soon.  loadInbox also primes the unread-count badge so
    // it shows even before the user opens the panel.
    loadPending();
    loadMemory();
    loadSettings();
    if (typeof loadInbox === 'function') loadInbox();
    if (typeof loadAgents === 'function') loadAgents();
    initItems(_currentProfile.id);
    return;
  }

  // Anonymous — figure out which sub-block to show.
  _currentProfile = null;
  destroyItems();
  _setTierTwoVisible(false);
  if (profiles.length === 0) {
    // No profiles enrolled at all — login is impossible until someone
    // enrols via the "Voice profiles" section.  Show needs-setup so at
    // least the message guides the user, but the dropdown will be empty.
    _populateProfileSelect(setupProfileSelect, profiles);
    _setAuthBlock('needs-setup');
    setupStatusEl.textContent = i18n('auth.record_profile_first');
    return;
  }

  // We have profiles.  Check if ANY of them has a passphrase set — if
  // yes, show login; if no, show setup.  We can't introspect bcrypt
  // hashes from the client, so heuristic: if every login attempt
  // immediately 404s, show setup.  Simpler: just show login by
  // default and let setup be reachable via the "Change password" path.
  _populateProfileSelect(authProfileSelect, profiles);
  _populateProfileSelect(setupProfileSelect, profiles);
  _setAuthBlock('anonymous');
  authStatusEl.textContent = '';
  setupStatusEl.textContent = '';
}

// ── Login / logout / setup ──────────────────────────────────────────────

async function doLogin() {
  const profileId = parseInt(authProfileSelect.value, 10);
  const passphrase = authPassphraseInput.value;
  if (!profileId || !passphrase) {
    authStatusEl.textContent = i18n('auth.select_profile_and_password');
    return;
  }
  authStatusEl.textContent = i18n('auth.checking');
  authLoginBtn.disabled = true;
  try {
    const r = await fetch('/api/auth/login', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile_id: profileId, passphrase }),
    });
    if (r.status === 401) {
      authStatusEl.textContent = i18n('auth.wrong_password');
      return;
    }
    if (!r.ok) {
      authStatusEl.textContent = i18n('auth.error_http', { status: r.status });
      return;
    }
    authPassphraseInput.value = '';
    authStatusEl.textContent = '';
    log(`auth: logged in as profile ${profileId}`);
    await loadAuthState();
  } catch (e) {
    authStatusEl.textContent = i18n('auth.network_error', { err: e.message });
  } finally {
    authLoginBtn.disabled = false;
  }
}

async function doLogout() {
  // Drop the server-side push subscription BEFORE clearing the cookie —
  // the DELETE endpoint requires auth and we need to address by
  // endpoint, which only the currently-logged-in tab can read from
  // pushManager.
  await _teardownWebPush();
  try {
    await fetch('/api/auth/logout', {
      method: 'POST',
      credentials: 'same-origin',
    });
  } catch (e) {
    log(`auth: logout failed: ${e.message}`);
  }
  log('auth: logged out');
  await loadAuthState();
}

async function doSetupPassphrase() {
  const profileId = parseInt(setupProfileSelect.value, 10);
  const passphrase = setupPassphraseEl.value;
  if (!profileId || !passphrase) {
    setupStatusEl.textContent = i18n('auth.select_and_new_password');
    return;
  }
  setupStatusEl.textContent = i18n('auth.creating');
  setupBtn.disabled = true;
  try {
    const r = await fetch('/api/auth/setup_passphrase', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile_id: profileId, passphrase }),
    });
    if (r.status === 403) {
      setupStatusEl.textContent = i18n('auth.already_has_password');
      return;
    }
    if (!r.ok) {
      setupStatusEl.textContent = i18n('auth.error_http', { status: r.status });
      return;
    }
    setupPassphraseEl.value = '';
    setupStatusEl.textContent = i18n('auth.done_now_login');
    await loadAuthState();
  } catch (e) {
    setupStatusEl.textContent = i18n('auth.network_error', { err: e.message });
  } finally {
    setupBtn.disabled = false;
  }
}

authLoginBtn?.addEventListener('click', doLogin);
authPassphraseInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') doLogin();
});
authLogoutBtn?.addEventListener('click', doLogout);
authSetPassBtn?.addEventListener('click', () => {
  // Switch the "known" block to a passphrase-set form by pre-selecting
  // the current profile in setupProfileSelect and toggling the visible
  // block.  Keeps the URL-less "rotate my password" flow inside one tab.
  if (_currentProfile) {
    setupProfileSelect.value = String(_currentProfile.id);
  }
  setupStatusEl.textContent = '';
  _setAuthBlock('needs-setup');
});
setupBtn?.addEventListener('click', doSetupPassphrase);

// ── Pending tab ─────────────────────────────────────────────────────────

async function loadPending() {
  if (!_currentProfile) return;
  pendingStatusEl.textContent = i18n('pending.loading');
  try {
    // include_recent=true asks the backend to also return the latest
    // executed / failed / rejected / expired actions (separate "recent"
    // array).  Capped at 10 — enough to glance at what just happened
    // without crowding the panel.
    const r = await fetch(
      `/api/users/${_currentProfile.id}/pending?include_recent=true&recent_limit=10`,
      { credentials: 'same-origin' },
    );
    if (!r.ok) {
      pendingStatusEl.textContent = `HTTP ${r.status}`;
      return;
    }
    const data = await r.json();
    renderPending(data.actions || []);
    renderPendingRecent(data.recent || []);
  } catch (e) {
    pendingStatusEl.textContent = i18n('pending.error', { err: e.message });
  }
}

function renderPending(actions) {
  pendingListEl.innerHTML = '';
  if (!actions.length) {
    pendingStatusEl.textContent = i18n('pending.none');
    pendingBadgeEl.style.display = 'none';
    return;
  }
  pendingStatusEl.textContent = '';
  pendingBadgeEl.textContent = String(actions.length);
  pendingBadgeEl.style.display = '';
  const now = Date.now() / 1000;
  for (const a of actions) {
    const ttl = a.expires_at - now;
    const ttlText = ttl < 3600
      ? i18n('pending.expires_in_min', { min: Math.max(1, Math.round(ttl / 60)) })
      : i18n('pending.valid_for_h', { h: Math.round(ttl / 3600) });
    const li = document.createElement('li');
    li.className = 'pending-card';
    li.innerHTML = `
      <div class="pending-card-summary">
        <span>${_escapeHtml(a.summary)}</span>
        <span class="pending-meta">${_escapeHtml(a.tool_name)} · ${ttlText}</span>
      </div>
      <button class="approve" type="button">${i18n('pending.approve')}</button>
      <button class="reject outline secondary" type="button">${i18n('pending.reject')}</button>`;
    li.querySelector('.approve').addEventListener('click', () => approvePending(a.id));
    li.querySelector('.reject').addEventListener('click', () => rejectPending(a.id));
    pendingListEl.appendChild(li);
  }
}

// "Recent" — recently finalised actions (executed / failed / rejected
// / expired).  Read-only.  Lives in #pending-recent-list which is
// created lazily so we don't need to touch index.html for it.
function renderPendingRecent(rows) {
  // Lazy-make the container so older HTML snapshots still work; tucks
  // it right after #pending-list.
  let recentEl = document.querySelector('#pending-recent-list');
  let recentHeader = document.querySelector('#pending-recent-header');
  if (!recentEl && pendingListEl?.parentElement) {
    recentHeader = document.createElement('small');
    recentHeader.id = 'pending-recent-header';
    recentHeader.style.color = 'var(--pico-muted-color)';
    recentHeader.style.marginTop = '0.5rem';
    recentHeader.style.display = 'block';
    recentEl = document.createElement('ul');
    recentEl.id = 'pending-recent-list';
    recentEl.style.padding = '0';
    recentEl.style.listStyle = 'none';
    recentEl.style.marginTop = '0.25rem';
    pendingListEl.insertAdjacentElement('afterend', recentHeader);
    recentHeader.insertAdjacentElement('afterend', recentEl);
  }
  if (!recentEl) return;
  recentEl.innerHTML = '';
  if (!rows.length) {
    if (recentHeader) recentHeader.textContent = '';
    return;
  }
  if (recentHeader) recentHeader.textContent = i18n('pending.recent_header');
  const STATUS_BADGES = {
    executed: '✓',
    execution_failed: '⚠',
    rejected: '✗',
    expired: '⏱',
  };
  const STATUS_LABEL = {
    executed:          i18n('pending.status.executed'),
    execution_failed:  i18n('pending.status.execution_failed'),
    rejected:          i18n('pending.status.rejected'),
    expired:           i18n('pending.status.expired'),
  };
  for (const r of rows) {
    const li = document.createElement('li');
    li.className = 'pending-card';
    // Same grid as live pending (1fr auto auto) but two trailing cells
    // are status text + small badge instead of action buttons.
    const badge = STATUS_BADGES[r.status] || '·';
    const label = STATUS_LABEL[r.status] || r.status || '';
    const whenS = r.approved_at || r.requested_at;
    let when = '';
    if (whenS) {
      const dt = new Date(whenS * 1000);
      when = dt.toLocaleString(undefined, { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' });
    }
    li.innerHTML = `
      <div class="pending-card-summary">
        <span>${badge} ${_escapeHtml(r.summary || r.tool_name || '?')}</span>
        <span class="pending-meta">${_escapeHtml(r.tool_name)} · ${_escapeHtml(label)}${when ? ' · ' + _escapeHtml(when) : ''}</span>
      </div>
      <span></span>
      <span></span>`;
    recentEl.appendChild(li);
  }
}

async function approvePending(actionId) {
  try {
    const r = await fetch(`/api/pending/${actionId}/approve`, {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!r.ok) {
      log(`pending: approve ${actionId} → HTTP ${r.status}`);
      return;
    }
    const data = await r.json();
    log(`pending: ✓ approved "${data.summary}"`);
    loadPending();
  } catch (e) {
    log(`pending: approve failed: ${e.message}`);
  }
}

async function rejectPending(actionId) {
  try {
    const r = await fetch(`/api/pending/${actionId}/reject`, {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!r.ok) {
      log(`pending: reject ${actionId} → HTTP ${r.status}`);
      return;
    }
    const data = await r.json();
    log(`pending: ✗ rejected "${data.summary}"`);
    loadPending();
  } catch (e) {
    log(`pending: reject failed: ${e.message}`);
  }
}

pendingRefreshEl?.addEventListener('click', loadPending);
// Auto-refresh when the user expands the panel — list goes stale fast
// (TTL is 24h, but new entries appear every voice turn).
pendingPanelEl?.addEventListener('toggle', () => {
  if (pendingPanelEl.open) loadPending();
});

// ── Inbox tab ───────────────────────────────────────────────────────────

const inboxPanelEl     = document.querySelector('#inbox-panel');
const inboxListEl      = document.querySelector('#inbox-list');
const inboxStatusEl    = document.querySelector('#inbox-status');
const inboxBadgeEl     = document.querySelector('#inbox-badge');
const inboxRefreshEl   = document.querySelector('#inbox-refresh');
const inboxUnreadOnlyEl = document.querySelector('#inbox-unread-only');
const inboxModeEl      = document.querySelector('#inbox-mode');

async function loadInbox() {
  if (!_currentProfile) return;
  if (inboxStatusEl) inboxStatusEl.textContent = i18n('inbox.loading');
  // Two view modes share this loader.  In Sent mode the unread filter
  // is meaningless (the sender doesn't have a listened flag on their
  // own outgoing rows) — we just hide the checkbox effect by ignoring
  // it.  The unread-count badge stays driven by the inbox endpoint so
  // a user reading their Sent panel still sees the inbox count tick.
  const mode = inboxModeEl?.value || 'inbox';
  const isSent = mode === 'sent';
  const unread = (!isSent && inboxUnreadOnlyEl?.checked) ? '&unread_only=true' : '';
  const url = isSent
    ? `/api/users/${_currentProfile.id}/outgoing_voicemail?limit=50`
    : `/api/users/${_currentProfile.id}/voicemail?limit=50${unread}`;
  try {
    const r = await fetch(url, { credentials: 'same-origin' });
    if (!r.ok) {
      if (inboxStatusEl) inboxStatusEl.textContent = `HTTP ${r.status}`;
      return;
    }
    const data = await r.json();
    // The Sent endpoint doesn't compute unread (no listened_at semantics
    // for the sender), so always refresh the badge via the inbox count
    // when in Sent mode; in Inbox mode the inbox endpoint already
    // returns it inline.
    if (isSent) {
      try {
        const r2 = await fetch(
          `/api/users/${_currentProfile.id}/voicemail?limit=1&unread_only=true`,
          { credentials: 'same-origin' },
        );
        if (r2.ok) {
          const d2 = await r2.json();
          _updateInboxBadge(d2.unread_count || 0);
        }
      } catch (e) { /* badge update is best-effort */ }
    } else {
      _updateInboxBadge(data.unread_count || 0);
    }
    renderInbox(data.messages || [], { mode });
  } catch (e) {
    if (inboxStatusEl) inboxStatusEl.textContent = `Error: ${e.message}`;
  }
}

function _updateInboxBadge(n) {
  if (!inboxBadgeEl) return;
  if (n > 0) {
    inboxBadgeEl.textContent = String(n);
    inboxBadgeEl.style.display = '';
  } else {
    inboxBadgeEl.style.display = 'none';
  }
}

function renderInbox(rows, opts = {}) {
  const mode = opts.mode || 'inbox';
  const isSent = mode === 'sent';
  if (!inboxListEl) return;
  inboxListEl.innerHTML = '';
  if (!rows.length) {
    if (inboxStatusEl) {
      inboxStatusEl.textContent = isSent
        ? i18n('inbox.outgoing_empty')
        : i18n('inbox.empty');
    }
    return;
  }
  if (inboxStatusEl) inboxStatusEl.textContent = '';
  for (const r of rows) {
    const li = document.createElement('li');
    li.className = 'inbox-card';
    li.style.padding = '0.5rem 0';
    li.style.borderBottom = '1px solid var(--pico-muted-border-color)';
    // Relative time for the header — same buckets as the inbox tool's
    // i18n.rel_time_* keys (just rendered client-side).
    const ageS = Math.max(0, Date.now() / 1000 - (r.created_at || 0));
    let when;
    if (ageS < 60) when = '<1m';
    else if (ageS < 3600) when = `${Math.round(ageS / 60)}m`;
    else if (ageS < 86400) when = `${Math.round(ageS / 3600)}h`;
    else when = `${Math.round(ageS / 86400)}d`;
    // Outgoing rows reverse the perspective: show «to RECIPIENT»
    // instead of «from SENDER», suppress the unread dot (the sender
    // doesn't own a listened flag on their own send), suppress the
    // reply input + delete button (sender can't mutate the
    // recipient's row), and surface the reply (or "no reply yet") as
    // the headline content under the original transcript.
    if (isSent) {
      const toLabel = r.to_name || '?';
      const replyLine = r.reply_text
        ? `<div style="margin:0.25rem 0; color:var(--pico-primary)">↳ ${_escapeHtml(r.reply_text)}</div>`
        : `<small style="color:var(--pico-muted-color); font-style:italic">${i18n('inbox.no_reply_yet')}</small>`;
      li.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:baseline; gap:0.5rem">
          <strong>${i18n('inbox.to')} ${_escapeHtml(toLabel)}</strong>
          <small style="color:var(--pico-muted-color)">${when}</small>
        </div>
        <div style="margin: 0.25rem 0; word-break: break-word">${_escapeHtml(r.transcript || '')}</div>
        ${replyLine}
        <audio controls preload="none" src="/api/voicemail/${r.id}/audio" style="width:100%; margin-top:0.25rem"></audio>
      `;
    } else {
      const fromLabel = r.from_name || i18n('inbox.guest');
      const unreadDot = r.listened_at ? '' : '<span style="color:var(--pico-primary);">●</span> ';
      li.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:baseline; gap:0.5rem">
          <strong>${unreadDot}${i18n('inbox.from')} ${_escapeHtml(fromLabel)}</strong>
          <small style="color:var(--pico-muted-color)">${when}</small>
        </div>
        <div style="margin: 0.25rem 0; word-break: break-word">${_escapeHtml(r.transcript || '')}</div>
        ${r.summary ? `<div style="margin:0.25rem 0; color:var(--pico-muted-color); font-style:italic">${_escapeHtml(r.summary)}</div>` : ''}
        <audio controls preload="none" src="/api/voicemail/${r.id}/audio" style="width:100%; margin-top:0.25rem"></audio>
        <div style="display:flex; gap:0.5rem; margin-top:0.5rem">
          <input class="reply-input" type="text" placeholder="${i18n('inbox.reply_placeholder')}" style="flex:1" ${r.reply_text ? `value="${_escapeHtml(r.reply_text)}"` : ''} />
          <button class="reply-send" type="button">${i18n('inbox.send')}</button>
          <button class="reply-delete outline secondary" type="button">${i18n('inbox.delete')}</button>
        </div>
        ${r.reply_text ? `<small style="color:var(--pico-muted-color); margin-top:0.25rem; display:block">↳ ${_escapeHtml(r.reply_text)}</small>` : ''}
      `;
    }
    // Inbox-only behaviours below — Sent rows don't expose the inputs.
    if (!isSent) {
      const audio = li.querySelector('audio');
      audio?.addEventListener('ended', async () => {
        try {
          await fetch(`/api/voicemail/${r.id}/listened`, {
            method: 'POST', credentials: 'same-origin',
          });
          loadInbox();
        } catch (e) { /* ignore */ }
      });
      li.querySelector('.reply-send')?.addEventListener('click', async () => {
        const input = li.querySelector('.reply-input');
        const text = (input?.value || '').trim();
        if (!text) return;
        try {
          const resp = await fetch(`/api/voicemail/${r.id}/reply`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ reply: text }),
          });
          if (!resp.ok) {
            log(`voicemail: reply ${r.id} → HTTP ${resp.status}`);
            return;
          }
          log(`voicemail: replied to id=${r.id}`);
          loadInbox();
        } catch (e) {
          log(`voicemail: reply failed: ${e.message}`);
        }
      });
      li.querySelector('.reply-delete')?.addEventListener('click', async () => {
        try {
          const resp = await fetch(`/api/voicemail/${r.id}`, {
            method: 'DELETE', credentials: 'same-origin',
          });
          if (!resp.ok) {
            log(`voicemail: delete ${r.id} → HTTP ${resp.status}`);
            return;
          }
          log(`voicemail: deleted id=${r.id}`);
          loadInbox();
        } catch (e) {
          log(`voicemail: delete failed: ${e.message}`);
        }
      });
    }
    inboxListEl.appendChild(li);
  }
}

inboxRefreshEl?.addEventListener('click', loadInbox);
inboxUnreadOnlyEl?.addEventListener('change', loadInbox);
inboxModeEl?.addEventListener('change', loadInbox);
inboxPanelEl?.addEventListener('toggle', () => {
  if (inboxPanelEl.open) loadInbox();
});

// ── Memory tab ──────────────────────────────────────────────────────────

async function loadMemory() {
  if (!_currentProfile) return;
  memoryStatusEl.textContent = i18n('memory.loading');
  try {
    const r = await fetch(
      `/api/users/${_currentProfile.id}/memory`,
      { credentials: 'same-origin' },
    );
    if (!r.ok) {
      memoryStatusEl.textContent = `HTTP ${r.status}`;
      return;
    }
    const data = await r.json();
    memoryEditorEl.value = data.memory || '';
    memoryStatusEl.textContent = '';
  } catch (e) {
    memoryStatusEl.textContent = i18n('memory.error', { err: e.message });
  }
}

async function saveMemory() {
  if (!_currentProfile) return;
  memoryStatusEl.textContent = i18n('memory.saving');
  memorySaveBtn.disabled = true;
  try {
    const r = await fetch(`/api/users/${_currentProfile.id}/memory`, {
      method: 'PUT',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: memoryEditorEl.value }),
    });
    if (!r.ok) {
      memoryStatusEl.textContent = `HTTP ${r.status}`;
      return;
    }
    memoryStatusEl.textContent = i18n('memory.saved');
    log(`memory: saved (${memoryEditorEl.value.length} chars)`);
  } catch (e) {
    memoryStatusEl.textContent = i18n('memory.error', { err: e.message });
  } finally {
    memorySaveBtn.disabled = false;
  }
}

memorySaveBtn?.addEventListener('click', saveMemory);
memoryReloadEl?.addEventListener('click', loadMemory);

// ── Settings tab ────────────────────────────────────────────────────────

async function loadSettings() {
  if (!_currentProfile) return;
  settingsStatusEl.textContent = i18n('settings.loading');
  try {
    const r = await fetch(
      `/api/users/${_currentProfile.id}/settings`,
      { credentials: 'same-origin' },
    );
    if (!r.ok) {
      settingsStatusEl.textContent = `HTTP ${r.status}`;
      return;
    }
    const data = await r.json();
    const s = data.settings || {};
    settingsLangEl.value = s.language || 'auto';
    settingsFormEl.value = s.formality || 'casual';
    settingsVoiceEl.value = s.tts_voice || '';
    settingsStyleEl.value = s.style_prompt || '';
    settingsRawEl.value = JSON.stringify(s, null, 2);
    settingsStatusEl.textContent = '';
    // Mirror the user's pinned language onto the UI right away.  "auto"
    // leaves the existing localStorage/navigator value untouched.  See
    // i18n.js: syncFromSettings handles the de/ru/en cases.
    syncFromSettings(s.language);
  } catch (e) {
    settingsStatusEl.textContent = i18n('settings.error', { err: e.message });
  }
}

async function saveSettings() {
  if (!_currentProfile) return;
  // Prefer the structured fields; if user has been hacking on the raw
  // JSON we parse that as the source of truth.  Validation lives on
  // the server (Pydantic) — we just ferry bytes.
  let body;
  try {
    body = JSON.parse(settingsRawEl.value);
  } catch (e) {
    settingsStatusEl.textContent = i18n('settings.json_broken', { err: e.message });
    return;
  }
  // Overlay structured fields on top of raw — they take precedence if
  // the user touched them in the dropdowns after the last reload.
  body.language    = settingsLangEl.value;
  body.formality   = settingsFormEl.value;
  body.tts_voice   = settingsVoiceEl.value || null;
  body.style_prompt = settingsStyleEl.value || null;
  settingsStatusEl.textContent = i18n('settings.saving');
  settingsSaveBtn.disabled = true;
  try {
    const r = await fetch(`/api/users/${_currentProfile.id}/settings`, {
      method: 'PUT',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => '');
      settingsStatusEl.textContent = `HTTP ${r.status}${txt ? `: ${txt}` : ''}`;
      return;
    }
    settingsStatusEl.textContent = i18n('settings.saved');
    log('settings: saved');
    // Re-paint from server's canonical version so we don't drift.
    loadSettings();
  } catch (e) {
    settingsStatusEl.textContent = i18n('settings.error', { err: e.message });
  } finally {
    settingsSaveBtn.disabled = false;
  }
}

settingsSaveBtn?.addEventListener('click', saveSettings);
settingsReloadBtn?.addEventListener('click', loadSettings);

// ── Connected devices (Wave 2 Phase 4 agents panel) ────────────────────
//
// Pulls /api/agents on auth-state changes and after the user expands the
// panel.  Read-only view — no add/remove UI here; the operator edits
// DESKTOP_AGENTS in docker-compose.yml and restarts.  We render the
// reachability dot + a row of capability badges so the user can see at
// a glance whether mail/screenshot are live on each device.

const agentsPanelEl  = document.querySelector('#agents-panel');
const agentsListEl   = document.querySelector('#agents-list');
const agentsStatusEl = document.querySelector('#agents-status');

// Last desktop-agent that handled a tool call.  Updated by the
// tool_called WS handler when its payload carries `target_agent`.
// Used by renderAgents() to mark that row with a small badge, and by
// _markAgentActive() to flash the row briefly when the event arrives
// while the panel is open (no full re-render — feels twitchy).
let _lastActiveAgent = { agent_id: null, ts: 0 };
const _AGENT_ACTIVE_WINDOW_MS = 60_000;  // badge stays for 1 min

function _markAgentActive(agentId) {
  _lastActiveAgent = { agent_id: agentId, ts: Date.now() };
  // If the panel is open, animate the matching row's strong-tag.
  // We don't re-render — that would re-mount the <audio> elements
  // in the inbox panel (sibling section) and stop any in-flight
  // voicemail playback.  CSS pulse via inline style is enough.
  if (!agentsListEl) return;
  for (const li of agentsListEl.children) {
    const strong = li.querySelector('strong');
    if (!strong) continue;
    if (strong.textContent.includes(agentId)) {
      strong.style.transition = 'background-color 0.6s ease-out';
      strong.style.backgroundColor = 'var(--pico-primary-background)';
      strong.style.borderRadius = '0.2rem';
      strong.style.padding = '0 0.2rem';
      setTimeout(() => {
        strong.style.backgroundColor = 'transparent';
      }, 600);
      break;
    }
  }
}

// Map capability flag → (emoji, i18n key).  Only the flags we care
// about show up as badges; the LLM-internal ones (e.g. `audit`) are
// intentionally omitted to keep the row terse.
const _CAP_BADGES = [
  { key: 'screenshot',           emoji: '📷', label: 'agents.cap_screenshot'    },
  { key: 'applescript',          emoji: '🍎', label: 'agents.cap_applescript'   },
  { key: 'pyautogui',            emoji: '🖱', label: 'agents.cap_pyautogui'     },
  { key: 'hotkey',               emoji: '⌨️', label: 'agents.cap_hotkey'        },
  { key: 'default_apps_resolver', emoji: '📂', label: 'agents.cap_default_apps' },
  { key: 'cursor_activity',      emoji: '👀', label: 'agents.cap_cursor'        },
];

function _platformLabel(p) {
  if (p === 'macos')   return i18n('agents.platform_macos');
  if (p === 'windows') return i18n('agents.platform_windows');
  if (p === 'linux')   return i18n('agents.platform_linux');
  return p || '?';
}

async function loadAgents() {
  if (!agentsListEl) return;
  try {
    const r = await fetch('/api/agents', { credentials: 'same-origin' });
    if (r.status === 401) {
      // Cookie expired or never present — leave the panel empty, the
      // auth banner already prompts for login.
      agentsListEl.innerHTML = '';
      if (agentsStatusEl) agentsStatusEl.textContent = i18n('agents.empty');
      return;
    }
    if (!r.ok) {
      if (agentsStatusEl) agentsStatusEl.textContent = `HTTP ${r.status}`;
      return;
    }
    const data = await r.json();
    renderAgents(data.agents || [], data.default || null);
  } catch (e) {
    if (agentsStatusEl) agentsStatusEl.textContent = `Error: ${e.message}`;
  }
}

function renderAgents(rows, defaultAgentId) {
  if (!agentsListEl) return;
  agentsListEl.innerHTML = '';
  if (!rows.length) {
    if (agentsStatusEl) agentsStatusEl.textContent = i18n('agents.empty');
    return;
  }
  if (agentsStatusEl) agentsStatusEl.textContent = '';
  const now = Date.now();
  for (const a of rows) {
    const li = document.createElement('li');
    li.style.padding = '0.4rem 0';
    li.style.borderBottom = '1px solid var(--pico-muted-border-color)';
    // Row 1: dot + agent_id + (platform) + default tag + active tag.
    const reachable = !!a.reachable;
    const dot = reachable ? '🟢' : '🔴';
    const status = i18n(reachable ? 'agents.online' : 'agents.offline');
    const platform = _platformLabel(a.platform);
    const isDefault = (a.agent_id === defaultAgentId) || a.default;
    const defaultTag = isDefault
      ? ` <small style="color:var(--pico-primary)">· ${_escapeHtml(i18n('agents.default_label'))}</small>`
      : '';
    // Show «active Ns ago» on the most-recently-used agent.  Resets
    // on every targeted tool call (see _markAgentActive).  The window
    // is 60s — long enough for the user to glance at the panel after
    // a voice turn, short enough to not pollute the panel forever.
    let activeTag = '';
    if (
      _lastActiveAgent.agent_id === a.agent_id
      && (now - _lastActiveAgent.ts) < _AGENT_ACTIVE_WINDOW_MS
    ) {
      const secs = Math.max(1, Math.round((now - _lastActiveAgent.ts) / 1000));
      activeTag = ` <small style="color:var(--pico-primary); font-weight:600">· ${_escapeHtml(i18n('agents.active_recent', { n: secs }))}</small>`;
    }
    // Row 2: capability badges.  Skip badges for flags that are false.
    const caps = a.capabilities || {};
    const badges = _CAP_BADGES
      .filter((b) => caps[b.key])
      .map((b) => `<span title="${_escapeHtml(i18n(b.label))}" style="margin-right:0.5rem">${b.emoji} ${_escapeHtml(i18n(b.label))}</span>`)
      .join('');
    li.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:baseline; gap:0.5rem">
        <strong>${dot} ${_escapeHtml(a.agent_id)}</strong>
        <small style="color:var(--pico-muted-color)">${_escapeHtml(platform)} · ${_escapeHtml(status)}${defaultTag}${activeTag}</small>
      </div>
      <div style="margin-top:0.25rem; color:var(--pico-muted-color); font-size:0.85rem">${badges || '<em style="color:var(--pico-muted-color)">—</em>'}</div>
    `;
    agentsListEl.appendChild(li);
  }
}

// Refresh on panel-open so the user sees fresh state without a manual
// reload.  Background poll updates the server-side cache every 30 s
// already; this just fetches it.
agentsPanelEl?.addEventListener('toggle', () => {
  if (agentsPanelEl.open) loadAgents();
});

// Initial auth state — populates banner + tier-2 sections if cookie is live.
loadAuthState();
