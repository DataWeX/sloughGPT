'use client'

import { useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { IconUpload } from '@/components/ui'

interface AudioCardProps {
  transcribing: boolean
  transcript: string | null
  synthesizing: boolean
  onTranscribe: (file: File) => void
  onSynthesize: (text: string) => void
}

export default function AudioCard({ transcribing, transcript, synthesizing, onTranscribe, onSynthesize }: AudioCardProps) {
  const audioInputRef = useRef<HTMLInputElement>(null)
  const [synthText, setSynthText] = useState('')

  const handleSynthesize = () => {
    if (synthText.trim()) onSynthesize(synthText.trim())
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
            <input ref={audioInputRef} type="file" accept="audio/*" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) onTranscribe(f) }} />
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
        </div>
      </CardContent>
    </Card>
  )
}
