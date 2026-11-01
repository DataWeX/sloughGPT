'use client'

import { useRef, useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { IconUpload } from '@sloughgpt/strui'

interface AudioCardProps {
  transcribing: boolean
  transcript: string | null
  synthesizing: boolean
  synthAudio?: { audio: string; duration_sec: number } | null
  onTranscribe: (file: File) => void
  onSynthesize: (text: string) => void
}

export default function AudioCard({ transcribing, transcript, synthesizing, synthAudio, onTranscribe, onSynthesize }: AudioCardProps) {
  const audioInputRef = useRef<HTMLInputElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [synthText, setSynthText] = useState('')

  useEffect(() => {
    if (synthAudio?.audio && audioRef.current) {
      audioRef.current.play().catch(() => /* autoplay blocked — user can click play */ {})
    }
  }, [synthAudio])

  const handleSynthesize = () => {
    if (synthText.trim()) onSynthesize(synthText.trim())
  }

  const handleDownload = () => {
    if (!synthAudio?.audio) return
    const a = document.createElement('a')
    a.href = synthAudio.audio
    a.download = `speech-${Date.now()}.wav`
    a.click()
  }

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Audio</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="text-xs font-medium text-muted-foreground mb-2">Speech-to-text</p>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="h-8 text-xs" onClick={() => audioInputRef.current?.click()} disabled={transcribing}>
              <IconUpload className="h-3.5 w-3.5 mr-1" />
              {transcribing ? 'Transcribing…' : 'Upload audio'}
            </Button>
            <input ref={audioInputRef} type="file" accept="audio/*" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) onTranscribe(f) }} aria-label="Upload audio" />
          </div>
          {transcript && (
            <div className="mt-2 p-2 rounded bg-muted/30 border border-border/40 text-xs text-muted-foreground">{transcript}</div>
          )}
        </div>
        <div>
          <p className="text-xs font-medium text-muted-foreground mb-2">Text-to-speech</p>
          <div className="flex items-center gap-2">
            <Input value={synthText} onChange={e => setSynthText(e.target.value)} placeholder="Text to speak…" className="h-8 text-xs flex-1" onKeyDown={e => { if (e.key === 'Enter') handleSynthesize() }} aria-label="Text to synthesize" />
            <Button size="sm" className="h-8 text-xs shrink-0" onClick={handleSynthesize} disabled={synthesizing || !synthText.trim()}>
              {synthesizing ? 'Synthesizing…' : 'Speak'}
            </Button>
          </div>
          {synthAudio && (
            <div className="mt-3 space-y-2">
              <audio ref={audioRef} src={synthAudio.audio} controls className="w-full h-8" />
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground">{synthAudio.duration_sec.toFixed(1)}s</span>
                <Button size="sm" variant="ghost" className="h-6 text-[10px] px-2" onClick={handleDownload}>
                  Download
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
