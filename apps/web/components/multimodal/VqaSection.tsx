'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input } from '@sloughgpt/strui'

interface VqaResult {
  answer: string
  question: string
  elapsed_ms: number
}

interface VqaSectionProps {
  asking: boolean
  onAsk: (file: File, question: string) => void
  answer?: VqaResult | null
}

export default function VqaSection({ asking, onAsk, answer }: VqaSectionProps) {
  const [file, setFile] = useState<File | null>(null)
  const [question, setQuestion] = useState('')

  const handleAsk = () => {
    if (file && question.trim()) onAsk(file, question.trim())
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Visual Question Answering</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <input
            type="file"
            accept="image/*"
            aria-label="VQA image"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="text-xs"
          />
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What is in this image?"
            className="h-8 text-xs flex-1"
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAsk()
            }}
            aria-label="Question"
          />
          <Button
            size="sm"
            className="h-8 text-xs shrink-0"
            onClick={handleAsk}
            disabled={asking || !file || !question.trim()}
          >
            {asking ? 'Asking…' : 'Ask'}
          </Button>
        </div>
        {answer && (
          <div
            className="rounded-lg border border-border/50 p-3 text-sm bg-muted/20"
            data-testid="vqa-answer"
          >
            <p>{answer.answer}</p>
            <p className="text-xs text-muted-foreground mt-1">{answer.elapsed_ms.toFixed(0)}ms</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
