'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

type RecordState = 'idle' | 'recording' | 'stopped'

interface Recording {
  blob: Blob
  url: string
  duration: number
  timestamp: string
}

export default function PronunciationRecording() {
  const [recordState, setRecordState] = useState<RecordState>('idle')
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [duration, setDuration] = useState(0)
  const [audioLevel, setAudioLevel] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animFrameRef = useRef<number | null>(null)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)

  const cleanup = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    if (animFrameRef.current) { cancelAnimationFrame(animFrameRef.current); animFrameRef.current = null }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  useEffect(() => () => cleanup(), [cleanup])

  const updateLevel = useCallback(() => {
    if (!analyserRef.current) return
    const data = new Uint8Array(analyserRef.current.fftSize)
    analyserRef.current.getByteTimeDomainData(data)
    let sum = 0
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128
      sum += v * v
    }
    setAudioLevel(Math.sqrt(sum / data.length))
    animFrameRef.current = requestAnimationFrame(updateLevel)
  }, [])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const audioCtx = new AudioContext()
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setRecordings(prev => [...prev, {
          blob,
          url,
          duration,
          timestamp: new Date().toISOString(),
        }])
        stream.getTracks().forEach(t => t.stop())
        audioCtx.close()
      }

      mediaRecorder.start()
      setRecordState('recording')
      setDuration(0)
      updateLevel()

      timerRef.current = setInterval(() => {
        setDuration(d => d + 100)
      }, 100)
    } catch {
      addToast('Microphone access denied', 'error')
    }
  }, [duration, updateLevel, addToast])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    cleanup()
    setRecordState('stopped')
    setAudioLevel(0)
  }, [cleanup])

  const resetRecording = useCallback(() => {
    setRecordState('idle')
    setDuration(0)
    setAudioLevel(0)
  }, [])

  const formatDuration = (ms: number) => {
    const s = Math.floor(ms / 1000)
    const m = Math.floor(s / 60)
    return `${m}:${(s % 60).toString().padStart(2, '0')}`
  }

  const deleteRecording = useCallback((index: number) => {
    setRecordings(prev => {
      const r = prev[index]
      if (r) URL.revokeObjectURL(r.url)
      return prev.filter((_, i) => i !== index)
    })
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Recording</span>
          <Badge variant="outline">{recordings.length} recordings</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col items-center gap-4">
          <div className="relative w-32 h-32 flex items-center justify-center">
            <div
              className={`absolute inset-0 rounded-full transition-all duration-200 ${
                recordState === 'recording'
                  ? 'bg-red-500/20 scale-110'
                  : 'bg-muted/30'
              }`}
              style={recordState === 'recording' ? {
                transform: `scale(${1 + audioLevel * 0.5})`,
              } : {}}
            />
            <div
              className={`w-20 h-20 rounded-full flex items-center justify-center cursor-pointer transition-colors ${
                recordState === 'recording'
                  ? 'bg-red-500 hover:bg-red-600'
                  : 'bg-primary hover:bg-primary/90'
              }`}
              onClick={recordState === 'recording' ? stopRecording : startRecording}
            >
              {recordState === 'recording' ? (
                <div className="w-6 h-6 bg-white rounded-sm" />
              ) : (
                <div className="w-8 h-8 bg-white rounded-full" />
              )}
            </div>
          </div>

          <div className="text-center">
            {recordState === 'recording' && (
              <div className="space-y-1">
                <p className="text-sm text-red-500 font-medium">Recording...</p>
                <p className="text-2xl font-mono">{formatDuration(duration)}</p>
                <div className="flex gap-0.5 justify-center h-4">
                  {Array.from({ length: 20 }).map((_, i) => (
                    <div
                      key={i}
                      className="w-1 bg-red-500 rounded-full transition-all duration-100"
                      style={{
                        height: `${Math.max(2, audioLevel * 40 * (1 - Math.abs(i - 10) / 10))}px`,
                      }}
                    />
                  ))}
                </div>
              </div>
            )}
            {recordState === 'idle' && (
              <p className="text-sm text-muted-foreground">Click to start recording</p>
            )}
            {recordState === 'stopped' && (
              <Button variant="ghost" size="sm" onClick={resetRecording}>
                <IconRefresh className="w-4 h-4 mr-1" />
                New Recording
              </Button>
            )}
          </div>
        </div>

        {recordings.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Recordings</p>
            {recordings.map((rec, i) => (
              <div key={i} className="flex items-center gap-3 p-2 rounded bg-muted/20">
                <audio controls src={rec.url} className="h-8 flex-1" />
                <span className="text-xs text-muted-foreground">{formatDuration(rec.duration)}</span>
                <Button variant="ghost" size="sm" onClick={() => deleteRecording(i)}>
                  ×
                </Button>
              </div>
            ))}
          </div>
        )}

        <p className="text-xs text-muted-foreground text-center">
          Record yourself pronouncing words and compare with the audio playback.
        </p>
      </CardContent>
    </Card>
  )
}
