'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { cn } from '@/lib/cn'
import { multimodalController } from '@/lib/controllers'
import { VoiceWaveform } from './VoiceWaveform'

import { Button } from '@/components/ui/button'

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
}

interface SpeechRecognitionResultList {
  length: number
  item(index: number): SpeechRecognitionResult
  [index: number]: SpeechRecognitionResult
}

interface SpeechRecognitionResult {
  length: number
  item(index: number): SpeechRecognitionAlternative
  [index: number]: SpeechRecognitionAlternative
  isFinal: boolean
}

interface SpeechRecognitionAlternative {
  transcript: string
  confidence: number
}

interface SpeechRecognition extends EventTarget {
  new (): SpeechRecognition
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onstart: (() => void) | null
  onend: (() => void) | null
  onerror: ((event: Event) => void) | null
  onresult: ((event: SpeechRecognitionEvent) => void) | null
}

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognition
    webkitSpeechRecognition?: SpeechRecognition
  }
}

interface VoiceInputProps {
  onTranscript: (text: string) => void
  disabled?: boolean
}

function MicrophoneIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
    </svg>
  )
}

function WaveformIcon({ className, isActive }: { className?: string; isActive: boolean }) {
  return (
    <svg className={cn(className, isActive && "animate-pulse")} fill="currentColor" viewBox="0 0 24 24" aria-hidden>
      <rect x="4" y="10" width="2" height="4" rx="1" className={isActive ? "animate-pulse" : ""} />
      <rect x="8" y="7" width="2" height="10" rx="1" className={isActive ? "animate-pulse" : ""} style={{ animationDelay: '100ms' }} />
      <rect x="12" y="4" width="2" height="16" rx="1" className={isActive ? "animate-pulse" : ""} style={{ animationDelay: '200ms' }} />
      <rect x="16" y="7" width="2" height="10" rx="1" className={isActive ? "animate-pulse" : ""} style={{ animationDelay: '300ms' }} />
      <rect x="20" y="10" width="2" height="4" rx="1" className={isActive ? "animate-pulse" : ""} style={{ animationDelay: '400ms' }} />
    </svg>
  )
}

export function VoiceInput({ onTranscript, disabled }: VoiceInputProps) {
  const [isListening, setIsListening] = useState(false)
  const [isBrowserSupported, setIsBrowserSupported] = useState(false)
  const [isServerSupported, setIsServerSupported] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    setIsBrowserSupported(typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window))
    multimodalController.getCapabilities().then(caps => {
      if (caps.speech_to_text) setIsServerSupported(true)
    }).catch(() => {})
  }, [])

  const startBrowserListening = useCallback(() => {
    const SRConstructor = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SRConstructor) return

    const recognition = new SRConstructor()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => setIsListening(true)
    recognition.onend = () => setIsListening(false)
    recognition.onerror = () => setIsListening(false)

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = Array.from(event.results)
        .map((result: SpeechRecognitionResult) => result[0].transcript)
        .join('')
      if (transcript) {
        onTranscript(transcript)
      }
    }

    recognition.start()
  }, [onTranscript])

  const startServerListening = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        setIsListening(false)
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        if (blob.size === 0) return
        const fd = new FormData()
        fd.append('file', blob, 'recording.webm')
        fd.append('language', 'en')
        try {
          const result = await multimodalController.transcribeAudio(blob as File)
          if (result.text) onTranscript(result.text)
        } catch {}
      }

      mediaRecorder.start()
      setIsListening(true)
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') mediaRecorder.stop()
      }, 5000)
    } catch {}
  }, [onTranscript])

  const toggleListening = useCallback(() => {
    if (isListening) {
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop()
      }
      setIsListening(false)
    } else if (isBrowserSupported) {
      startBrowserListening()
    } else if (isServerSupported) {
      startServerListening()
    }
  }, [isListening, isBrowserSupported, isServerSupported, startBrowserListening, startServerListening])

  if (!isBrowserSupported && !isServerSupported) return null

  return (
    <>
      <span className="sr-only" aria-live="assertive" aria-atomic="true">
        {isListening ? 'Listening…' : ''}
      </span>
      {isListening && (
        <div className="hidden sm:flex items-center px-1" aria-hidden="true">
          <VoiceWaveform
            level={0.6}
            bars={12}
            variant="mic"
            width={48}
            height={20}
          />
        </div>
      )}
      <Button
        variant="ghost"
        size="icon"
        onClick={toggleListening}
        disabled={disabled}
        className={cn(
          "h-10 w-10 transition-all duration-200",
          isListening
            ? "text-destructive hover:text-destructive"
            : "text-muted-foreground",
          disabled && "opacity-50 cursor-not-allowed"
        )}
        aria-label={isListening ? "Stop listening" : "Start voice input"}
        aria-pressed={isListening}
        title={`Voice input${isBrowserSupported ? '' : ' (server)'}`}
      >
        {isListening ? (
          <WaveformIcon className="h-5 w-5" isActive={true} />
        ) : (
          <MicrophoneIcon className="h-5 w-5" />
        )}
      </Button>
    </>
  )
}
