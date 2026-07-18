'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { chatController } from '@/lib/chat-controller'
import { voiceController } from '@/lib/voice-controller'

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

// ── Speech Recognition Types (browser API) �────────────────────────────

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
  onresult: ((e: any) => void) | null
}

// ── Constants ─────────────────────────────────────────────────────────

const SILENCE_TIMEOUT_MS = 2000
const AUTO_RESUME_DELAY_MS = 400
const INTERRUPT_CHECK_MS = 100

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

function createAudioLevelMonitor(): {
  start: (stream: MediaStream) => void
  stop: () => void
  getLevel: () => number
  onLevel: ((level: number) => void) | null
} {
  let analyser: AnalyserNode | null = null
  let source: MediaStreamAudioSourceNode | null = null
  let ctx: AudioContext | null = null
  let raf: number | null = null
  let level = 0

  return {
    onLevel: null,
    getLevel: () => level,
    start(stream: MediaStream) {
      try {
        ctx = new AudioContext()
        analyser = ctx.createAnalyser()
        analyser.fftSize = 256
        analyser.smoothingTimeConstant = 0.4
        source = ctx.createMediaStreamSource(stream)
        source.connect(analyser)

        const dataArray = new Uint8Array(analyser.frequencyBinCount)
        const tick = () => {
          if (!analyser) return
          analyser.getByteFrequencyData(dataArray)
          let sum = 0
          for (let i = 0; i < dataArray.length; i++) sum += dataArray[i]
          level = sum / dataArray.length / 255
          this.onLevel?.(level)
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
  const recognitionRef = useRef<SRInstance | null>(null)
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const finalTextRef = useRef('')
  const stateRef = useRef<VoiceState>('idle')
  const onMessageRef = useRef(onMessage)
  const onExchangeRef = useRef(onExchange)
  const conversationRef = useRef<VoiceExchange[]>([])
  const abortControllerRef = useRef<AbortController | null>(null)
  const speakingRef = useRef(false)
  const audioMonitorRef = useRef(createAudioLevelMonitor())
  const micStreamRef = useRef<MediaStream | null>(null)
  const interruptCheckRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const settingsRef = useRef(settings)

  // Keep settings ref in sync
  useEffect(() => { settingsRef.current = settings }, [settings])

  // Keep refs in sync
  useEffect(() => { onMessageRef.current = onMessage }, [onMessage])
  useEffect(() => { onExchangeRef.current = onExchange }, [onExchange])
  useEffect(() => { stateRef.current = state }, [state])

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
      audioMonitorRef.current.onLevel = (level) => setMicLevel(level)
      audioMonitorRef.current.start(stream)
    } catch {
      // Mic not available — visual-only mode
    }
  }, [])

  const stopMic = useCallback(() => {
    audioMonitorRef.current.stop()
    audioMonitorRef.current.onLevel = null
    setMicLevel(0)
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(t => t.stop())
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
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) {
      setErrorMessage('Speech recognition not supported — try Chrome or Safari')
      setState('error')
      return
    }

    setFinalText('')
    finalTextRef.current = ''
    setInterimText('')
    setResponseText('')
    setErrorMessage(null)

    await startMic()

    const recognition: SRInstance = new SR()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => {
      setState('listening')
    }

    recognition.onend = () => {
      if (stateRef.current === 'listening') {
        setState('idle')
      }
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
    }

    recognition.onerror = (e: Event) => {
      const errType = (e as { error?: string })?.error || 'unknown'
      if (errType === 'not-allowed') {
        setErrorMessage('Microphone access denied')
      } else if (errType !== 'aborted') {
        setErrorMessage(`Speech error: ${errType}`)
      }
      if (errType !== 'aborted') {
        setState('error')
      }
    }

    recognition.onresult = (event: any) => {
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
      if (final) {
        const newFinal = finalTextRef.current + final
        finalTextRef.current = newFinal
        setFinalText(newFinal)
        setInterimText('')
        resetSilenceTimer(newFinal)
      }
      if (interim) {
        setInterimText(interim)
        resetSilenceTimer(finalTextRef.current + interim)
      }
    }

    recognitionRef.current = recognition
    recognition.start()
  }, [resetSilenceTimer, startMic])

  // ── Stop Listening ─────────────────────────────────────────────────

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
    }
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
    stopMic()
    if (stateRef.current === 'listening') {
      setState('idle')
    }
  }, [stopMic])

  // ── Speak response aloud (with interrupt support) ───────────────────

  const speakResponse = useCallback(async (text: string): Promise<void> => {
    speakingRef.current = true
    startInterruptDetection()

    try {
      // Try server-side TTS first
      const result = await voiceController.tts(text)
      if (result.backend === 'hf-model' && result.audio) {
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
          const voice = window.speechSynthesis.getVoices().find(v => v.name === voiceName)
          if (voice) utterance.voice = voice
        }
        utterance.onend = () => resolve()
        utterance.onerror = () => resolve()
        window.speechSynthesis.speak(utterance)
      })
    }

    speakingRef.current = false
    stopInterruptDetection()
  }, [startInterruptDetection, stopInterruptDetection])

  // ── Handle Submit (text → AI → speak) ──────────────────────────────

  const handleSubmit = useCallback(async (text: string) => {
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
      let sentenceQueue: string[] = []
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
        setErrorMessage(e.message || 'Generation failed')
        setState('error')
      }
    } finally {
      abortControllerRef.current = null
    }
  }, [startListening, speakResponse])

  // ── Push-to-talk (hold spacebar to listen) ────────────────────────

  const pushToTalkActiveRef = useRef(false)

  useEffect(() => {
    if (!settings.pushToTalk) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.code !== 'Space' || e.repeat) return

      e.preventDefault()
      if (!pushToTalkActiveRef.current && (stateRef.current === 'idle' || stateRef.current === 'error')) {
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
    setSettingsState(prev => ({ ...prev, ...partial }))
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
