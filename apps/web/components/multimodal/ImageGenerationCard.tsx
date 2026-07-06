'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'

interface ImageGenerationCardProps {
  generating: boolean
  onGenerate: (prompt: string) => void
  generatedImage?: string | null
}

export default function ImageGenerationCard({ generating, onGenerate, generatedImage }: ImageGenerationCardProps) {
  const [genPrompt, setGenPrompt] = useState('')

  const handleGenerate = () => {
    if (genPrompt.trim()) onGenerate(genPrompt.trim())
  }

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Image Generation</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <Input value={genPrompt} onChange={e => setGenPrompt(e.target.value)} placeholder="A cat in a spacesuit…" className="h-8 text-xs flex-1" onKeyDown={e => { if (e.key === 'Enter') handleGenerate() }} aria-label="Image generation prompt" />
          <Button size="sm" className="h-8 text-xs shrink-0" onClick={handleGenerate} disabled={generating || !genPrompt.trim()}>
            {generating ? 'Generating…' : 'Generate'}
          </Button>
        </div>
        {generatedImage && (
          <div className="rounded-lg border border-border/50 overflow-hidden">
            <img src={generatedImage} alt="Generated" className="w-full h-auto max-h-64 object-contain bg-muted/20" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
