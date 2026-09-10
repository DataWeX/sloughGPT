'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { chatDB } from '@/lib/db'

interface Recording {
  id: string
  blob: Blob
  url: string
  duration: number
  label: string
  timestamp: number
}

const STORAGE_KEY = 'sloughgpt-voice-recordings'

async function loadRecordings(): Promise<Recording[]> {
  try {
    const entry = await chatDB.getKV<Recording[]>(STORAGE_KEY)
    if (entry && Array.isArray(entry)) return entry
  } catch { /* corrupted */ }
  return []
}

async function saveRecordings(recordings: Recording[]) {
  try { await chatDB.setKV(STORAGE_KEY, recordings) } catch { /* quota exceeded */ }
}

export function VoiceRecordingCard() {
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [recordState, setRecordState] = useState<'idle' | 'recording' | 'paused'>('idle')
  const [duration, setDuration] = useState(0)
  const [audioLevel, setAudioLevel] = useState(0)
  const [label, setLabel] = useState('')
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animFrameRef = useRef<number | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    loadRecordings().then(setRecordings)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    }
  }, [])

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
      streamRef.current = stream
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
        const id = `rec-${Date.now()}`
        const entry: Recording = {
          id,
          blob,
          url,
          duration,
          label: label || `Recording ${recordings.length + 1}`,
          timestamp: Date.now(),
        }
        const updated = [...recordings, entry]
        setRecordings(updated)
        saveRecordings(updated).catch(() => {})
        stream.getTracks().forEach(t => t.stop())
        audioCtx.close()
        setLabel('')
      }

      mediaRecorder.start()
      setRecordState('recording')
      setDuration(0)
      updateLevel()

      timerRef.current = setInterval(() => {
        setDuration(d => d + 100)
      }, 100)
    } catch {
      console.error('Microphone access denied')
    }
  }, [duration, updateLevel, label, recordings])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    if (animFrameRef.current) { cancelAnimationFrame(animFrameRef.current); animFrameRef.current = null }
    setRecordState('idle')
    setAudioLevel(0)
  }, [])

  const deleteRecording = useCallback((id: string) => {
    const rec = recordings.find(r => r.id === id)
    if (rec) URL.revokeObjectURL(rec.url)
    const updated = recordings.filter(r => r.id !== id)
    setRecordings(updated)
    saveRecordings(updated).catch(() => {})
  }, [recordings])

  const playRecording = useCallback((rec: Recording) => {
    const audio = new Audio(rec.url)
    audio.play()
  }, [])

  const formatDuration = (ms: number) => {
    const s = Math.floor(ms / 1000)
    const m = Math.floor(s / 60)
    return `${m}:${(s % 60).toString().padStart(2, '0')}`
  }

  return (
    <Card data-testid="voice-recording">
      <CardHeader>
        <CardTitle className="text-base">Voice Recording</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col items-center gap-3">
          <div
            className={`w-16 h-16 rounded-full flex items-center justify-center cursor-pointer transition-all ${
              recordState === 'recording'
                ? 'bg-red-500 hover:bg-red-600 scale-110'
                : 'bg-primary hover:bg-primary/90'
            }`}
            onClick={recordState === 'recording' ? stopRecording : startRecording}
          >
            {recordState === 'recording' ? (
              <div className="w-5 h-5 bg-white rounded-sm" />
            ) : (
              <div className="w-6 h-6 bg-white rounded-full" />
            )}
          </div>

          {recordState === 'recording' && (
            <div className="text-center space-y-2">
              <p className="text-sm text-red-500 font-medium">Recording... {formatDuration(duration)}</p>
              <div className="flex gap-0.5 justify-center h-6">
                {Array.from({ length: 24 }).map((_, i) => (
                  <div
                    key={i}
                    className="w-0.5 bg-red-500 rounded-full transition-all duration-75"
                    style={{
                      height: `${Math.max(2, audioLevel * 50 * (1 - Math.abs(i - 12) / 12))}px`,
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {recordings.length > 0 && (
          <div className="space-y-1.5">
            {recordings.map(rec => (
              <div key={rec.id} className="flex items-center gap-2 p-2 rounded border border-border/60 text-sm group hover:bg-muted/50 transition-colors">
                <button
                  className="text-primary hover:text-primary/80 text-xs font-medium"
                  onClick={() => playRecording(rec)}
                >
                  Play
                </button>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">{rec.label}</p>
                  <p className="text-[10px] text-muted-foreground">{formatDuration(rec.duration)}</p>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  className="opacity-0 group-hover:opacity-100 transition-opacity text-destructive"
                  onClick={() => deleteRecording(rec.id)}
                >
                  Del
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
