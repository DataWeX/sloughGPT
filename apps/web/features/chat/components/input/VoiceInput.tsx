'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { multimodalController } from '@/lib/controllers'
import { extractErrorMessage } from '@/lib/error-utils'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'
import { VoiceWaveform } from './VoiceWaveform'

import { cn, Button, IconMic, AudioWaveform } from '@sloughgpt/strui'

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
  onAudioRecorded?: (blob: Blob) => void
  onSend?: () => void
  disabled?: boolean
}

function MicrophoneIcon({ className }: { className?: string }) {
  return <IconMic className={className} />
}

function WaveformIcon({ className, isActive }: { className?: string; isActive: boolean }) {
  return <AudioWaveform className={className} isActive={isActive} />
}

export function VoiceInput({ onTranscript, onAudioRecorded, onSend, disabled }: VoiceInputProps) {
  const [isListening, setIsListening] = useState(false)
  const [isBrowserSupported, setIsBrowserSupported] = useState(false)
  const [isServerSupported, setIsServerSupported] = useState(false)
  const addToast = useToastStore(s => s.addToast)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    let active = true
    setIsBrowserSupported('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)
    multimodalController.getCapabilities().then(caps => {
      if (active && caps.speech_to_text) setIsServerSupported(true)
    }).catch(() => /* voice capabilities unavailable — browser-only mode */ {})
    return () => { active = false }
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
    recognition.onerror = (e: Event) => {
      setIsListening(false)
      const err = (e as { error?: string }).error
      if (err === 'not-allowed') addToast('Microphone access denied by browser', 'error')
      else if (err === 'no-speech') addToast('No speech detected — try again', 'info')
      else if (err === 'network') addToast('Speech recognition network error', 'error')
      else if (err && err !== 'aborted') addToast(`Voice input error: ${err}`, 'error')
    }

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

        // Pass audio blob to parent for voice message saving
        if (onAudioRecorded) {
          onAudioRecorded(blob)
        }

        try {
          const result = await multimodalController.transcribeAudio(blob as File)
          if (result.text) onTranscript(result.text)
        } catch (err) {
          const msg = extractErrorMessage(err)
          if (msg.includes('501') || msg.includes('not available')) {
            addToast('Speech recognition not available — use Chrome or Edge for browser voice input', 'error')
          } else {
            addToast('Could not audio transcription', 'error')
          }
          logger.error('Could not audio transcription', { exception: String(err) })
        }
      }

      mediaRecorder.start()
      setIsListening(true)
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') mediaRecorder.stop()
      }, 5000)
    } catch (err) {
      const msg = extractErrorMessage(err)
      if (msg.includes('NotAllowedError') || msg.includes('Permission denied')) {
        addToast('Microphone access denied — allow microphone in browser settings', 'error')
      } else {
        addToast('Could not access microphone', 'error')
      }
      logger.error('Could not microphone access', { exception: String(err) })
    }
  }, [onTranscript, onAudioRecorded])

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

  const supported = isBrowserSupported || isServerSupported

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
        disabled={disabled || !supported}
        className={cn(
          "h-10 w-10 transition-all duration-200",
          isListening
            ? "text-destructive hover:text-destructive"
            : "text-muted-foreground",
          (disabled || !supported) && "opacity-50 cursor-not-allowed"
        )}
        aria-label={isListening ? "Stop listening" : supported ? "Start voice input" : "Voice input unavailable — use Chrome or Edge"}
        aria-pressed={isListening}
        title={!supported ? 'Voice input requires Chrome, Edge, or Safari' : isListening ? 'Stop listening' : 'Voice input'}
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
