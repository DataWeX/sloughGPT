'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { chatController } from '@/lib/chat-controller'
import { voiceController } from '@/lib/voice-controller'
import { createSpeechEngine, type SpeechEngine } from '../voice/speech-engine'

// ── Types ──────────────────────────────────────────────────────────────

export type VoiceState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error'

export interface VoiceExchange {
  id: string
  userText: string
  assistantText: string
  timestamp: number
}

export interface VoiceChatCallbacks {
  onMessage: (text: string) => void
  onExchange?: (exchange: VoiceExchange) => void
}

// ── Speech Recognition Types ───────────────────────────────────────────
// Browser-SR types moved to ../voice/speech-engine.ts (owner of the STT
// pipeline; useVoiceChat now drives a SpeechEngine, not a raw SR instance).

// ── Constants ─────────────────────────────────────────────────────────

const SILENCE_TIMEOUT_MS = 2000
const AUTO_RESUME_DELAY_MS = 400
const INTERRUPT_CHECK_MS = 100
/**
 * `micLevel` state is a UI-only, quantized signal (0.1 buckets) — emit it at
 * most ~8x/s. The per-frame value lives on `micLevelRef`, which visuals sample
 * on their own rAF loop, so React never re-renders the voice overlay at
 * audio-frame rate.
 */
const MIC_LEVEL_EMIT_MS = 120
const MIC_LEVEL_BUCKETS = 10

/**
 * Per-frame audio features for visuals (the center-stage warble). Published on
 * the same two-channel rule as `micLevelRef`: the monitor writes every frame,
 * visuals sample on their own rAF loop, React never re-renders at audio rate.
 */
export interface VoiceAudioFeatures {
  /** 0–1 smoothed RMS loudness */
  intensity: number
  /** Fundamental frequency in Hz; 0 while unvoiced/noisy */
  pitch: number
  /** 0–1 normalized spectral centroid (brightness) */
  timbre: number
}

const ZERO_FEATURES: VoiceAudioFeatures = { intensity: 0, pitch: 0, timbre: 0 }

// ── Voice Settings ────────────────────────────────────────────────────

export interface VoiceSettings {
  /** Speech rate for browser TTS (0.5–2.0) */
  rate: number
  /** Pitch for browser TTS (0.5–2.0) */
  pitch: number
  /** Mic level threshold to trigger interrupt (0.05–0.5) */
  interruptThreshold: number
  /** Auto-resume listening after AI finishes speaking */
  autoResume: boolean
  /** Preferred voice name (browser TTS) */
  voiceName: string | null
  /** Push-to-talk mode: hold spacebar to listen */
  pushToTalk: boolean
  /** Speak sentences as they complete (don't wait for full response) */
  streamingTTS: boolean
}

const DEFAULT_SETTINGS: VoiceSettings = {
  rate: 0.95,
  pitch: 1.0,
  interruptThreshold: 0.15,
  autoResume: true,
  voiceName: null,
  pushToTalk: false,
  streamingTTS: true,
}

// ── Audio Level Monitor ───────────────────────────────────────────────

/**
 * Fundamental-frequency detection via normalized autocorrelation over the
 * byte time-domain window. Returns 0 when the frame is unvoiced or noisy —
 * the caller treats that as "hold the last pitch".
 */
function detectPitch(buf: Uint8Array, sampleRate: number): number {
  const n = buf.length
  const minLag = Math.floor(sampleRate / 400) // human voice ceiling: 400 Hz
  const maxLag = Math.min(Math.floor(sampleRate / 70), n - 1) // floor: 70 Hz
  if (maxLag <= minLag) return 0
  let bestLag = 0
  let bestCorr = 0
  for (let lag = minLag; lag <= maxLag; lag += 2) {
    let corr = 0
    let energy = 0
    for (let i = 0; i + lag < n; i += 2) {
      const a = buf[i] - 128
      const b = buf[i + lag] - 128
      corr += a * b
      energy += a * a
    }
    if (energy === 0) continue
    const norm = corr / energy // 0..1 — period-matched frames score high
    if (norm > bestCorr) {
      bestCorr = norm
      bestLag = lag
    }
  }
  if (bestCorr < 0.3 || !bestLag) return 0 // unvoiced: no confident period
  return sampleRate / bestLag
}

function createAudioLevelMonitor(): {
  start: (stream: MediaStream) => void
  stop: () => void
  getLevel: () => number
  onLevel: ((level: number) => void) | null
  onFeatures: ((features: VoiceAudioFeatures) => void) | null
} {
  let analyser: AnalyserNode | null = null
  let source: MediaStreamAudioSourceNode | null = null
  let ctx: AudioContext | null = null
  let raf: number | null = null
  let level = 0

  return {
    onLevel: null,
    onFeatures: null,
    getLevel: () => level,
    start(stream: MediaStream) {
      try {
        ctx = new AudioContext()
        analyser = ctx.createAnalyser()
        // 1024-sample window: the 70 Hz pitch floor needs a ~630-sample lag,
        // which the old 256 fftSize could not hold. `level` stays the mean of
        // the same 0–Nyquist spectrum (bin count scales with fftSize), so
        // interrupt detection semantics are unchanged.
        analyser.fftSize = 1024
        analyser.smoothingTimeConstant = 0.4
        source = ctx.createMediaStreamSource(stream)
        source.connect(analyser)

        const sampleRate = ctx.sampleRate
        const freqData = new Uint8Array(analyser.frequencyBinCount)
        const timeData = new Uint8Array(analyser.fftSize)
        const binHz = sampleRate / analyser.fftSize
        let intensity = 0
        let pitch = 0
        let timbre = 0

        const tick = () => {
          if (!analyser) return
          analyser.getByteFrequencyData(freqData)
          let sum = 0
          for (let i = 0; i < freqData.length; i++) sum += freqData[i]
          level = sum / freqData.length / 255
          this.onLevel?.(level)

          if (this.onFeatures) {
            analyser.getByteTimeDomainData(timeData)

            // Intensity: RMS around the 128 silence midpoint, lightly smoothed.
            let sq = 0
            for (let i = 0; i < timeData.length; i++) {
              const v = (timeData[i] - 128) / 128
              sq += v * v
            }
            const rms = Math.min(1, Math.sqrt(sq / timeData.length) * 4)
            intensity += (rms - intensity) * 0.3

            // Timbre: spectral centroid mapped 0..1 over 0–8 kHz.
            let num = 0
            let den = 0
            for (let i = 0; i < freqData.length; i++) {
              const m = freqData[i]
              if (m > 16) {
                num += i * m
                den += m
              }
            }
            const centroid = den > 0 ? Math.min(1, ((num / den) * binHz) / 8000) : 0
            timbre += (centroid - timbre) * 0.1

            // Pitch: EMA over voiced frames; detectPitch's 0 decays the
            // estimate toward silence instead of jumping.
            pitch += (detectPitch(timeData, sampleRate) - pitch) * 0.25

            this.onFeatures?.({ intensity, pitch, timbre })
          }
          raf = requestAnimationFrame(tick)
        }
        tick()
      } catch {
        // Web Audio not available
      }
    },
    stop() {
      if (raf !== null) cancelAnimationFrame(raf)
      raf = null
      source?.disconnect()
      analyser = null
      source = null
      ctx?.close()
      ctx = null
      level = 0
    },
  }
}

// ── Hook ───────────────────────────────────────────────────────────────

export function useVoiceChat({ onMessage, onExchange }: VoiceChatCallbacks) {
  const [state, setState] = useState<VoiceState>('idle')
  const [interimText, setInterimText] = useState('')
  const [finalText, setFinalText] = useState('')
  const [responseText, setResponseText] = useState('')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [conversation, setConversation] = useState<VoiceExchange[]>([])
  const [micLevel, setMicLevel] = useState(0)
  const [settings, setSettingsState] = useState<VoiceSettings>(DEFAULT_SETTINGS)
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([])

  // Load available browser voices
  useEffect(() => {
    const loadVoices = () => {
      if ('speechSynthesis' in window) {
        setAvailableVoices(window.speechSynthesis.getVoices())
      }
    }
    loadVoices()
    window.speechSynthesis?.addEventListener('voiceschanged', loadVoices)
    return () => window.speechSynthesis?.removeEventListener('voiceschanged', loadVoices)
  }, [])

  // Refs for imperative access inside callbacks
  const engineRef = useRef<SpeechEngine | null>(null)
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const finalTextRef = useRef('')
  const stateRef = useRef<VoiceState>('idle')
  const onMessageRef = useRef(onMessage)
  const onExchangeRef = useRef(onExchange)
  const conversationRef = useRef<VoiceExchange[]>([])
  const abortControllerRef = useRef<AbortController | null>(null)
  const speakingRef = useRef(false)
  const audioMonitorRef = useRef(createAudioLevelMonitor())
  /** Per-frame mic level (0–1) — read imperatively by visuals, never rendered from. */
  const micLevelRef = useRef(0)
  /** Per-frame features (intensity/pitch/timbre) — the warble's signal, same ref channel. */
  const audioFeaturesRef = useRef<VoiceAudioFeatures>({ ...ZERO_FEATURES })
  const lastMicLevelEmitRef = useRef(0)
  const micStreamRef = useRef<MediaStream | null>(null)
  const interruptCheckRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const settingsRef = useRef(settings)

  // Keep settings ref in sync
  useEffect(() => {
    settingsRef.current = settings
  }, [settings])

  // Keep refs in sync
  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])
  useEffect(() => {
    onExchangeRef.current = onExchange
  }, [onExchange])
  useEffect(() => {
    stateRef.current = state
  }, [state])

  // ── Interrupt detection ─────────────────────────────────────────────

  const startInterruptDetection = useCallback(() => {
    if (interruptCheckRef.current) return
    interruptCheckRef.current = setInterval(() => {
      const level = audioMonitorRef.current.getLevel()
      if (level > settingsRef.current.interruptThreshold && speakingRef.current) {
        // User is speaking — interrupt AI
        speakingRef.current = false
        window.speechSynthesis?.cancel()
        abortControllerRef.current?.abort()
        abortControllerRef.current = null
        // Resume listening after interrupt
        setState('idle')
        setTimeout(() => {
          if (stateRef.current === 'idle') {
            startListening()
          }
        }, AUTO_RESUME_DELAY_MS)
      }
    }, INTERRUPT_CHECK_MS)
  }, [])

  const stopInterruptDetection = useCallback(() => {
    if (interruptCheckRef.current) {
      clearInterval(interruptCheckRef.current)
      interruptCheckRef.current = null
    }
  }, [])

  // ── Start microphone + audio level monitoring ───────────────────────

  const startMic = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      micStreamRef.current = stream
      audioMonitorRef.current.onLevel = (level) => {
        micLevelRef.current = level
        const now = Date.now()
        if (now - lastMicLevelEmitRef.current < MIC_LEVEL_EMIT_MS) return
        lastMicLevelEmitRef.current = now
        const bucketed = Math.round(level * MIC_LEVEL_BUCKETS) / MIC_LEVEL_BUCKETS
        setMicLevel((prev) => (prev === bucketed ? prev : bucketed))
      }
      audioMonitorRef.current.onFeatures = (features) => {
        audioFeaturesRef.current = features
      }
      audioMonitorRef.current.start(stream)
    } catch {
      // Mic not available — visual-only mode
    }
  }, [])

  const stopMic = useCallback(() => {
    audioMonitorRef.current.stop()
    audioMonitorRef.current.onLevel = null
    audioMonitorRef.current.onFeatures = null
    micLevelRef.current = 0
    audioFeaturesRef.current = { ...ZERO_FEATURES }
    setMicLevel(0)
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop())
      micStreamRef.current = null
    }
  }, [])

  // ── Silence timer ──────────────────────────────────────────────────

  const resetSilenceTimer = useCallback((text: string) => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
    silenceTimerRef.current = setTimeout(() => {
      if (stateRef.current === 'listening' && text.trim()) {
        stopListening()
        handleSubmit(text.trim())
      }
    }, SILENCE_TIMEOUT_MS)
  }, [])

  // ── Start Listening ────────────────────────────────────────────────

  const startListening = useCallback(async () => {
    setFinalText('')
    finalTextRef.current = ''
    setInterimText('')
    setResponseText('')
    setErrorMessage(null)

    await startMic()

    // SpeechEngine picks the recognizer: Web Speech API when available,
    // server transcription (mic → PCM → /multimodal/transcribe) otherwise —
    // so voice chat works in Electron panes (OpenCode desktop) and Android
    // WebView instead of dead-ending at "not supported".
    const engine = createSpeechEngine({
      onStart: () => setState('listening'),
      onFinal: (text) => {
        const newFinal = finalTextRef.current + text
        finalTextRef.current = newFinal
        setFinalText(newFinal)
        setInterimText('')
        resetSilenceTimer(newFinal)
      },
      onInterim: (text) => {
        setInterimText(text)
        resetSilenceTimer(finalTextRef.current + text)
      },
      onError: (message, fatal) => {
        setErrorMessage(message)
        if (fatal) setState('error')
      },
      onEnd: () => {
        if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
        if (stateRef.current === 'listening') setState('idle')
      },
    })
    engineRef.current = engine
    // Reuse the monitor's stream so server mode never opens a second mic.
    engine.start(micStreamRef.current)
  }, [resetSilenceTimer, startMic])

  // ── Stop Listening ─────────────────────────────────────────────────

  const stopListening = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.stop()
      engineRef.current = null
    }
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
    stopMic()
    if (stateRef.current === 'listening') {
      setState('idle')
    }
  }, [stopMic])

  // ── Speak response aloud (with interrupt support) ───────────────────

  const speakResponse = useCallback(
    async (text: string): Promise<void> => {
      speakingRef.current = true
      startInterruptDetection()

      try {
        // Try server-side TTS first
        const result = await voiceController.tts(text)
        if (result.audio && result.backend !== 'browser-fallback') {
          await voiceController.playAudio(result.audio, result.sample_rate)
          speakingRef.current = false
          stopInterruptDetection()
          return
        }
      } catch {
        // Fall through to browser TTS
      }

      // Browser speechSynthesis fallback — speak in sentences for interruptibility
      if (!('speechSynthesis' in window)) {
        speakingRef.current = false
        stopInterruptDetection()
        return
      }

      const sentences = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [text]
      const { rate, pitch, voiceName } = settingsRef.current
      for (const sentence of sentences) {
        if (!speakingRef.current) break // interrupted
        await new Promise<void>((resolve) => {
          const utterance = new SpeechSynthesisUtterance(sentence.trim())
          utterance.rate = rate
          utterance.pitch = pitch
          // Select voice by name if specified
          if (voiceName) {
            const voice = window.speechSynthesis.getVoices().find((v) => v.name === voiceName)
            if (voice) utterance.voice = voice
          }
          utterance.onend = () => resolve()
          utterance.onerror = () => resolve()
          window.speechSynthesis.speak(utterance)
        })
      }

      speakingRef.current = false
      stopInterruptDetection()
    },
    [startInterruptDetection, stopInterruptDetection],
  )

  // ── Handle Submit (text → AI → speak) ──────────────────────────────

  const handleSubmit = useCallback(
    async (text: string) => {
      if (!text) return
      setState('processing')
      setResponseText('')
      setErrorMessage(null)

      onMessageRef.current(text)

      // Create abort controller for interruptible generation
      const controller = new AbortController()
      abortControllerRef.current = controller

      try {
        let fullResponse = ''
        let sentenceBuffer = ''
        const sentenceQueue: string[] = []
        let speaking = false

        const flushSentence = async (sentence: string) => {
          if (!sentence.trim()) return
          if (settingsRef.current.streamingTTS && !speakingRef.current) {
            speaking = true
            setState('speaking')
            await speakResponse(sentence.trim())
            speaking = false
            if (stateRef.current === 'speaking') setState('processing')
          }
        }

        for await (const token of chatController.stream(text)) {
          fullResponse += token
          setResponseText(fullResponse)

          if (settingsRef.current.streamingTTS) {
            // Accumulate tokens and detect sentence boundaries
            sentenceBuffer += token
            const sentenceMatch = sentenceBuffer.match(/^[^.!?]*[.!?]\s*/)
            if (sentenceMatch) {
              const sentence = sentenceMatch[0]
              sentenceBuffer = sentenceBuffer.slice(sentence.length)
              await flushSentence(sentence)
            }
          }
        }

        // Speak any remaining text in the buffer
        if (settingsRef.current.streamingTTS && sentenceBuffer.trim()) {
          await flushSentence(sentenceBuffer)
        }

        // Record exchange
        const exchange: VoiceExchange = {
          id: `voice-${Date.now()}`,
          userText: text,
          assistantText: fullResponse,
          timestamp: Date.now(),
        }
        conversationRef.current = [...conversationRef.current, exchange]
        setConversation([...conversationRef.current])
        onExchangeRef.current?.(exchange)

        // If streaming TTS already spoke everything, just finish
        if (settingsRef.current.streamingTTS && speakingRef.current) {
          // Wait for any in-progress speech to finish
        } else if (fullResponse && fullResponse.length > 1) {
          // Non-streaming: speak full response
          setState('speaking')
          await speakResponse(fullResponse)
        }

        // Auto-resume listening
        if (settingsRef.current.autoResume) {
          setState('idle')
          setTimeout(() => startListening(), AUTO_RESUME_DELAY_MS)
        } else {
          setState('idle')
        }
      } catch (e: unknown) {
        if (e instanceof Error && e.name !== 'AbortError') {
          setErrorMessage(e.message || 'Could not generation')
          setState('error')
        }
      } finally {
        abortControllerRef.current = null
      }
    },
    [startListening, speakResponse],
  )

  // ── Push-to-talk (hold spacebar to listen) ────────────────────────

  const pushToTalkActiveRef = useRef(false)

  useEffect(() => {
    if (!settings.pushToTalk) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.code !== 'Space' || e.repeat) return

      e.preventDefault()
      if (
        !pushToTalkActiveRef.current &&
        (stateRef.current === 'idle' || stateRef.current === 'error')
      ) {
        pushToTalkActiveRef.current = true
        startListening()
      }
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.code !== 'Space') return

      e.preventDefault()
      if (pushToTalkActiveRef.current) {
        pushToTalkActiveRef.current = false
        if (stateRef.current === 'listening') {
          stopListening()
          const text = finalTextRef.current.trim()
          if (text) handleSubmit(text)
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [settings.pushToTalk, startListening, stopListening, handleSubmit])

  // ── Toggle ─────────────────────────────────────────────────────────

  const handleToggle = useCallback(() => {
    if (state === 'listening') {
      stopListening()
    } else if (state === 'idle' || state === 'error') {
      startListening()
    }
    // processing/speaking: toggle interrupts
    if (state === 'speaking') {
      speakingRef.current = false
      window.speechSynthesis?.cancel()
      abortControllerRef.current?.abort()
      setState('idle')
      setTimeout(() => startListening(), AUTO_RESUME_DELAY_MS)
    }
  }, [state, startListening, stopListening])

  // ── Settings update ─────────────────────────────────────────────────

  const updateSettings = useCallback((partial: Partial<VoiceSettings>) => {
    setSettingsState((prev) => ({ ...prev, ...partial }))
  }, [])

  // ── Cleanup on unmount ────────────────────────────────────────────

  useEffect(() => {
    return () => {
      stopListening()
      stopInterruptDetection()
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel()
      }
      abortControllerRef.current?.abort()
    }
  }, [stopListening, stopInterruptDetection])

  return {
    state,
    interimText,
    finalText,
    responseText,
    errorMessage,
    conversation,
    micLevel,
    micLevelRef,
    audioFeaturesRef,
    settings,
    availableVoices,
    updateSettings,
    startListening,
    stopListening,
    handleToggle,
    // Convenience flags
    isListening: state === 'listening',
    isProcessing: state === 'processing',
    isSpeaking: state === 'speaking',
  }
}
