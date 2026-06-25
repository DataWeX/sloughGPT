'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { chatController } from '@/lib/chat-controller'
import { voiceController } from '@/lib/voice-controller'

// ── Types ──────────────────────────────────────────────────────────────

export type VoiceState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error'

export interface VoiceChatCallbacks {
  onMessage: (text: string) => void
}

// ── Speech Recognition Types (browser API) ─────────────────────────────

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

const SILENCE_TIMEOUT_MS = 2000
const AUTO_RESUME_DELAY_MS = 400

// ── Hook ───────────────────────────────────────────────────────────────

export function useVoiceChat({ onMessage }: VoiceChatCallbacks) {
  const [state, setState] = useState<VoiceState>('idle')
  const [interimText, setInterimText] = useState('')
  const [finalText, setFinalText] = useState('')
  const [responseText, setResponseText] = useState('')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // Refs for imperative access inside callbacks
  const recognitionRef = useRef<SRInstance | null>(null)
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const finalTextRef = useRef('')
  const stateRef = useRef<VoiceState>('idle')
  const onMessageRef = useRef(onMessage)

  // Keep callback ref in sync
  useEffect(() => { onMessageRef.current = onMessage }, [onMessage])

  // Sync stateRef
  useEffect(() => { stateRef.current = state }, [state])

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

  const startListening = useCallback(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
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
      const errType = (e as any)?.error || 'unknown'
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
  }, [resetSilenceTimer])

  // ── Stop Listening ─────────────────────────────────────────────────

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
    }
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
    if (stateRef.current === 'listening') {
      setState('idle')
    }
  }, [])

  // ── Speak response aloud ───────────────────────────────────────────

  const speakResponse = useCallback(async (text: string): Promise<void> => {
    // Try server-side TTS first
    try {
      const result = await voiceController.tts(text)
      if (result.backend === 'hf-model' && result.audio) {
        await voiceController.playAudio(result.audio, result.sample_rate)
        return
      }
    } catch {
      // Fall through to browser TTS
    }

    // Browser speechSynthesis fallback
    if (!('speechSynthesis' in window)) return
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 0.95
      utterance.pitch = 1.0
      utterance.onend = () => resolve()
      utterance.onerror = () => resolve()
      window.speechSynthesis.speak(utterance)
    })
  }, [])

  // ── Handle Submit (text → AI → speak) ──────────────────────────────

  const handleSubmit = useCallback(async (text: string) => {
    if (!text) return
    setState('processing')
    setResponseText('')
    setErrorMessage(null)

    onMessageRef.current(text)

    try {
      let fullResponse = ''
      for await (const token of chatController.stream(text)) {
        fullResponse += token
        setResponseText(fullResponse)
      }

      if (fullResponse && fullResponse.length > 1) {
        setState('speaking')
        await speakResponse(fullResponse)
      }

      // Auto-resume listening
      setState('idle')
      setTimeout(() => startListening(), AUTO_RESUME_DELAY_MS)
    } catch (e: any) {
      setErrorMessage(e.message || 'Generation failed')
      setState('error')
    }
  }, [startListening, speakResponse])

  // ── Toggle ─────────────────────────────────────────────────────────

  const handleToggle = useCallback(() => {
    if (state === 'listening') {
      stopListening()
    } else if (state === 'idle' || state === 'error') {
      startListening()
    }
    // processing/speaking: ignore toggle
  }, [state, startListening, stopListening])

  // ── Cleanup on unmount ────────────────────────────────────────────

  useEffect(() => {
    return () => {
      stopListening()
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current)
      // Cancel any speech in progress
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel()
      }
    }
  }, [stopListening])

  return {
    state,
    interimText,
    finalText,
    responseText,
    errorMessage,
    startListening,
    stopListening,
    handleToggle,
    // Convenience flags
    isListening: state === 'listening',
    isProcessing: state === 'processing',
    isSpeaking: state === 'speaking',
  }
}
