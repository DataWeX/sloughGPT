'use client'

/**
 * Speech Engine — pluggable STT for voice chat (card 631e6181).
 *
 * Two recognizers behind one surface:
 *
 *   browser — Web Speech API (SpeechRecognition). Best UX (live interim
 *             results), but Chrome/Safari only: it phones Google's speech
 *             servers. Dead in Electron panes (OpenCode desktop) and
 *             Android WebView — the constructor is absent, or present-but
 *             failing with `network` (Electron ships no Google API keys).
 *
 *   server  — mic capture → 16 kHz mono s16le PCM → POST
 *             /multimodal/transcribe → domain VoiceEngine (vosk offline /
 *             speech_recognition fallback). Works in every browser with mic
 *             access. The PCM contract matches the server's real one:
 *             ServerSpeechRecognizer wraps bytes as
 *             AudioData(data, 16000, 2) / KaldiRecognizer(model, 16000) —
 *             raw PCM, no container parsing (a MediaRecorder webm blob
 *             would decode as garbage).
 *
 * Failover: browser is preferred; on a fatal browser error (not-allowed,
 * network, service-not-allowed, audio-capture) the engine transparently
 * switches to server mode exactly once. `aborted` is not an error. When
 * both recognizers are unavailable (no mic), onError(..., fatal=true).
 */

import { voiceController } from '@/lib/voice-controller'

export type SpeechEngineMode = 'browser' | 'server'

export interface SpeechEngineHandlers {
  /** Recognition actually started (hook maps this to state 'listening'). */
  onStart?: () => void
  /** Finalized transcript chunk — same semantics as SR final results. */
  onFinal?: (text: string) => void
  /** In-progress transcript (browser mode only). */
  onInterim?: (text: string) => void
  /** Error message; fatal=true means the engine is spent (state 'error'). */
  onError?: (message: string, fatal: boolean) => void
  /** Recognition ended (hook maps this to state 'idle'). */
  onEnd?: () => void
  /** Active recognizer (initial selection or failover). */
  onMode?: (mode: SpeechEngineMode) => void
}

export interface SpeechEngine {
  /** Start listening. Reuses `stream` in server mode when provided. */
  start(stream?: MediaStream | null): void
  /** Stop; flushes any captured speech for transcription first. */
  stop(): void
  /** Stop immediately; drops captured audio (no transcribe). */
  abort(): void
  readonly mode: SpeechEngineMode | null
}

// ── Browser SR types (moved here from useVoiceChat.ts) ─────────────────

export interface SpeechRecognitionEventLike extends Event {
  resultIndex: number
  results: SpeechRecognitionResultListLike
}

export interface SpeechRecognitionResultListLike {
  length: number
  [index: number]: SpeechRecognitionResultLike
}

export interface SpeechRecognitionResultLike {
  isFinal: boolean
  length: number
  [index: number]: SpeechRecognitionAlternativeLike
}

export interface SpeechRecognitionAlternativeLike {
  transcript: string
  confidence: number
}

interface SRInstance {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onstart: (() => void) | null
  onend: (() => void) | null
  onerror: ((e: Event) => void) | null
  onresult: ((e: SpeechRecognitionEventLike) => void) | null
}

// ── Server-mode capture constants ──────────────────────────────────────

/** RMS above this counts as speech (matches interrupt-detection scale). */
const SPEECH_THRESHOLD = 0.01
/** Silence after speech that ends an utterance (samples, @ capture rate). */
const END_SILENCE_MS = 1200
/** Hard cap per utterance — flushes a continuously-noisy room. */
const MAX_UTTERANCE_MS = 15000
/** ScriptProcessor buffer — 4096 @ 48 kHz ≈ 85 ms per audio process tick. */
const PROCESSOR_SIZE = 4096
/** The server contract rate (vosk KaldiRecognizer / AudioData wrapper). */
const TARGET_RATE = 16000
/** Ignore captured audio shorter than this (bytes of s16le @ 16 kHz). */
const MIN_PCM_BYTES = Math.floor(TARGET_RATE * 0.3) * 2
const SR_LANG = 'en-US'

/** SR errors where the browser pipeline itself is dead — fail over. */
const FATAL_SR_ERRORS = new Set(['not-allowed', 'network', 'service-not-allowed', 'audio-capture'])

// ── PCM conversion ─────────────────────────────────────────────────────

/** Float32 [-1,1] → Int16 LE PCM, linear-resampled to 16 kHz. */
export function floatToPcm16(input: Float32Array, sourceRate: number): Int16Array {
  if (sourceRate === TARGET_RATE) {
    const out = new Int16Array(input.length)
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]))
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
    }
    return out
  }
  const ratio = sourceRate / TARGET_RATE
  const outLen = Math.floor(input.length / ratio)
  const out = new Int16Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio
    const i0 = Math.floor(pos)
    const i1 = Math.min(i0 + 1, input.length - 1)
    const frac = pos - i0
    const s = Math.max(-1, Math.min(1, input[i0] * (1 - frac) + input[i1] * frac))
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  return out
}

// ── Engine ─────────────────────────────────────────────────────────────

export function createSpeechEngine(handlers: SpeechEngineHandlers): SpeechEngine {
  let mode: SpeechEngineMode | null = null
  let sr: SRInstance | null = null
  let failedOver = false
  /** stop() called — no new capture; a flush-on-stop still delivers its final. */
  let stopped = false
  /** abort() called — drop captured audio, suppress late callbacks. */
  let aborted = false

  // Server-mode capture state
  let audioCtx: AudioContext | null = null
  let sourceNode: MediaStreamAudioSourceNode | null = null
  let processorNode: ScriptProcessorNode | null = null
  let sinkNode: GainNode | null = null
  let captureStream: MediaStream | null = null
  let ownsStream = false
  let chunks: Float32Array[] = []
  let speaking = false
  let utteranceSamples = 0
  let silenceSamples = 0
  /** Serializes transcribe calls — one utterance in flight at a time. */
  let transcribeQueue: Promise<void> = Promise.resolve()

  const teardownServer = (flush: boolean) => {
    if (flush && audioCtx) flushUtterance()
    try {
      processorNode?.disconnect()
      sourceNode?.disconnect()
      sinkNode?.disconnect()
      void audioCtx?.close()
    } catch {
      // context already closed
    }
    processorNode = null
    sourceNode = null
    sinkNode = null
    audioCtx = null
    if (ownsStream && captureStream) {
      captureStream.getTracks().forEach((t) => t.stop())
    }
    captureStream = null
    ownsStream = false
    chunks = []
    speaking = false
    utteranceSamples = 0
    silenceSamples = 0
  }

  const teardownBrowser = () => {
    const s = sr
    sr = null
    if (s) {
      s.onstart = null
      s.onend = null
      s.onerror = null
      s.onresult = null
      try {
        s.abort()
      } catch {
        // already stopped
      }
    }
  }

  const flushUtterance = () => {
    const captured = chunks
    chunks = []
    speaking = false
    utteranceSamples = 0
    silenceSamples = 0
    const total = captured.reduce((n, c) => n + c.length, 0)
    if (total === 0) return
    const merged = new Float32Array(total)
    let off = 0
    for (const c of captured) {
      merged.set(c, off)
      off += c.length
    }
    const rate = audioCtx?.sampleRate ?? TARGET_RATE
    const pcm = floatToPcm16(merged, rate)
    if (pcm.byteLength < MIN_PCM_BYTES) return
    transcribeQueue = transcribeQueue.then(async () => {
      try {
        const blob = new Blob([pcm.buffer as ArrayBuffer], { type: 'audio/l16' })
        const result = await voiceController.transcribe(blob)
        if (result.text && !aborted) handlers.onFinal?.(result.text)
      } catch {
        if (!aborted && !stopped) handlers.onError?.('Transcription failed', false)
      }
    })
  }

  const startServer = async (stream?: MediaStream | null) => {
    if (stopped || aborted) return
    mode = 'server'
    handlers.onMode?.('server')
    try {
      if (stream) {
        captureStream = stream
        ownsStream = false
      } else {
        captureStream = await navigator.mediaDevices.getUserMedia({ audio: true })
        ownsStream = true
      }
    } catch {
      if (aborted) return
      handlers.onError?.('Microphone access denied', true)
      handlers.onEnd?.()
      return
    }
    if (stopped || aborted) {
      // stop()/abort() landed while getUserMedia was in flight — release the
      // stream we now own instead of leaking a live capture.
      teardownServer(false)
      return
    }
    try {
      audioCtx = new AudioContext({ sampleRate: TARGET_RATE })
    } catch {
      // Older Safari rejects the sampleRate option — fall back and resample.
      audioCtx = new AudioContext()
    }
    const ctx = audioCtx
    sourceNode = ctx.createMediaStreamSource(captureStream)
    // ScriptProcessor is deprecated but universal; AudioWorklet needs async
    // module loading and this fallback path must work in bare webviews.
    processorNode = ctx.createScriptProcessor(PROCESSOR_SIZE, 1, 1)
    // Route through a muted gain so the processor runs without mic feedback.
    sinkNode = ctx.createGain()
    sinkNode.gain.value = 0

    const endSilenceSamples = Math.floor((END_SILENCE_MS / 1000) * ctx.sampleRate)
    const maxUtteranceSamples = Math.floor((MAX_UTTERANCE_MS / 1000) * ctx.sampleRate)

    processorNode.onaudioprocess = (e: AudioProcessingEvent) => {
      if (stopped || aborted || !audioCtx) return
      const data = e.inputBuffer.getChannelData(0)
      let sum = 0
      for (let i = 0; i < data.length; i++) sum += data[i] * data[i]
      const rms = Math.sqrt(sum / data.length)

      if (!speaking) {
        if (rms > SPEECH_THRESHOLD) {
          speaking = true
          utteranceSamples = 0
          silenceSamples = 0
          chunks = [new Float32Array(data)]
        }
        // Pre-speech audio is discarded — nothing to keep.
        return
      }

      chunks.push(new Float32Array(data))
      utteranceSamples += data.length
      if (rms > SPEECH_THRESHOLD) {
        silenceSamples = 0
      } else {
        silenceSamples += data.length
      }
      if (utteranceSamples >= maxUtteranceSamples || silenceSamples >= endSilenceSamples) {
        flushUtterance()
      }
    }

    sourceNode.connect(processorNode)
    processorNode.connect(sinkNode)
    sinkNode.connect(ctx.destination)
    if (stopped || aborted) {
      teardownServer(false)
      return
    }
    handlers.onStart?.()
  }

  const startBrowser = (SR: NonNullable<typeof window.SpeechRecognition>) => {
    const recognition: SRInstance = new SR()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = SR_LANG

    recognition.onstart = () => {
      handlers.onStart?.()
    }

    recognition.onend = () => {
      if (sr === recognition) handlers.onEnd?.()
    }

    recognition.onerror = (e: Event) => {
      const errType = (e as { error?: string })?.error || 'unknown'
      if (errType === 'aborted') return
      if (FATAL_SR_ERRORS.has(errType) && !failedOver) {
        failedOver = true
        teardownBrowser()
        // Server mode reports its own fatal error if the mic is truly denied.
        void startServer(lastStream)
        return
      }
      if (FATAL_SR_ERRORS.has(errType)) {
        handlers.onError?.(
          errType === 'not-allowed' ? 'Microphone access denied' : `Speech error: ${errType}`,
          true,
        )
        return
      }
      // Non-fatal (e.g. no-speech): keep listening, surface the message.
      handlers.onError?.(`Speech error: ${errType}`, false)
    }

    recognition.onresult = (event: SpeechRecognitionEventLike) => {
      let interim = ''
      let final = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          final += result[0].transcript
        } else {
          interim += result[0].transcript
        }
      }
      if (final) handlers.onFinal?.(final)
      if (interim) handlers.onInterim?.(interim)
    }

    sr = recognition
    recognition.start()
  }

  /** Stream handed to start() — reused if we ever fail over to server mode. */
  let lastStream: MediaStream | null = null

  return {
    get mode() {
      return mode
    },
    start(stream?: MediaStream | null) {
      stopped = false
      aborted = false
      lastStream = stream ?? null
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      if (SR) {
        mode = 'browser'
        handlers.onMode?.('browser')
        startBrowser(SR)
        return
      }
      void startServer(stream)
    },
    stop() {
      stopped = true
      if (sr) {
        const s = sr
        sr = null
        s.onend = null // hook gets onEnd exactly once, below
        try {
          s.stop()
        } catch {
          // already stopped
        }
        handlers.onEnd?.()
        return
      }
      // Flush-on-stop: captured speech still gets transcribed (aborted is
      // what suppresses the final, not stopped).
      if (audioCtx) teardownServer(true)
      handlers.onEnd?.()
    },
    abort() {
      aborted = true
      if (sr) {
        teardownBrowser()
        handlers.onEnd?.()
        return
      }
      if (audioCtx) {
        teardownServer(false)
        handlers.onEnd?.()
      }
    },
  }
}
