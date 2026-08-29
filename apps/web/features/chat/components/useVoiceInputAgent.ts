'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { estimateTokens } from '@/lib/format-bytes'

export interface AudioStats {
  duration: number
  words: number
  tokens: number
  estimatedCost: number
}

export const MODEL_PRICING: Record<string, { input: number; output: number }> = {
  'gpt-4': { input: 0.03, output: 0.06 },
  'gpt-4-turbo': { input: 0.01, output: 0.03 },
  'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
  'whisper-1': { input: 0.006, output: 0 },
}

export interface UseVoiceInputAgentReturn {
  isRecording: boolean
  isPaused: boolean
  duration: number
  transcript: string
  audioStats: AudioStats | null
  showStats: boolean
  error: string | null
  startRecording: () => Promise<void>
  stopRecording: () => void
  pauseRecording: () => void
  resumeRecording: () => void
  handleSendTranscript: () => void
  formatDuration: (seconds: number) => string
  setShowStats: (v: boolean) => void
}

export function useVoiceInputAgent(
  onTranscript: (text: string) => void,
  onSend?: () => void,
  model: string = 'whisper-1'
): UseVoiceInputAgentReturn {
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [duration, setDuration] = useState(0)
  const [transcript, setTranscript] = useState('')
  const [audioStats, setAudioStats] = useState<AudioStats | null>(null)
  const [showStats, setShowStats] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const startTimeRef = useRef<number>(0)

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const audioUrl = URL.createObjectURL(blob)
        
        // Calculate stats
        const durationSec = duration
        const words = Math.floor(durationSec * 2.5) // ~150 words per minute
        const tokens = estimateTokens(Array(words).fill('word').join(' '))
        const pricing = MODEL_PRICING[model] || MODEL_PRICING['whisper-1']
        const estimatedCost = (durationSec / 60) * pricing.input

        setAudioStats({
          duration: durationSec,
          words,
          tokens,
          estimatedCost,
        })

        // Create audio element for playback
        const audio = new Audio(audioUrl)
        audio.onended = () => URL.revokeObjectURL(audioUrl)
      }

      mediaRecorder.start(1000) // Collect data every second
      setIsRecording(true)
      setIsPaused(false)
      setDuration(0)
      setTranscript('')
      setError(null)
      setAudioStats(null)
      startTimeRef.current = Date.now()

      // Start duration timer
      timerRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 1000)
    } catch (err) {
      setError('Microphone access denied')
    }
  }, [duration, model])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop())
    }
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    setIsRecording(false)
    setIsPaused(false)
  }, [])

  const pauseRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.pause()
      setIsPaused(true)
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  const resumeRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'paused') {
      mediaRecorderRef.current.resume()
      setIsPaused(false)
      startTimeRef.current = Date.now() - duration * 1000
      timerRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 1000)
    }
  }, [duration])

  const handleSendTranscript = useCallback(() => {
    if (transcript.trim()) {
      onTranscript(transcript.trim())
      if (onSend) onSend()
      setTranscript('')
    }
  }, [transcript, onTranscript, onSend])

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return {
    isRecording, isPaused, duration, transcript, audioStats,
    showStats, error,
    startRecording, stopRecording, pauseRecording, resumeRecording,
    handleSendTranscript, formatDuration, setShowStats,
  }
}
