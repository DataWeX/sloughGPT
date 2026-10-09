'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Textarea, Label } from '@sloughgpt/strui'

interface EmbedResult {
  embedding: number[]
  dimensions: number
  model: string
}

interface InferEmbedCardProps {
  prompt: string
  result: EmbedResult | null
  loading: boolean
  onPromptChange: (v: string) => void
  onRun: () => void
}

export function InferEmbedCard({
  prompt,
  result,
  loading,
  onPromptChange,
  onRun,
}: InferEmbedCardProps) {
  const barValues = result?.embedding.slice(0, 20) ?? []
  const maxVal = Math.max(...barValues.map(Math.abs), 1)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Text Embedding</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <Label htmlFor="embed-prompt">Prompt</Label>
          <Textarea
            id="embed-prompt"
            value={prompt}
            onChange={e => onPromptChange(e.target.value)}
            placeholder="Enter text to embed..."
            className="h-24 text-sm resize-none"
            aria-label="Embed prompt"
          />
        </div>

        <Button onClick={onRun} disabled={loading || !prompt.trim()} className="w-full">
          {loading ? 'Embedding...' : 'Run'}
        </Button>

        {result && (
          <div className="space-y-2">
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>{result.dimensions} dimensions</span>
              <span>{result.model}</span>
            </div>
            <div className="flex items-end gap-px h-16" aria-label="Embedding visualization">
              {barValues.map((v, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t"
                  style={{
                    height: `${Math.abs(v) / maxVal * 100}%`,
                    backgroundColor: v >= 0 ? 'hsl(var(--primary))' : 'hsl(var(--destructive))',
                    opacity: 0.7,
                  }}
                />
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
