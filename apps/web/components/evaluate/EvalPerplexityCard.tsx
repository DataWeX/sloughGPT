'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Textarea } from '@sloughgpt/strui'
import { apiPost } from '@/lib/http-client'
import { useToastStore } from '@/lib/toast-store'

interface PerplexityResult {
  perplexity: number
  loss: number
  tokens: number
}

interface EvalPerplexityCardProps {
  result?: PerplexityResult | null
  onResult?: (result: PerplexityResult) => void
}

export function EvalPerplexityCard({ result: externalResult, onResult }: EvalPerplexityCardProps) {
  const addToast = useToastStore(s => s.addToast)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [internalResult, setInternalResult] = useState<PerplexityResult | null>(null)

  const result = externalResult ?? internalResult

  const handleCalculate = async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const data = await apiPost<PerplexityResult>('/benchmark/perplexity', { text })
      setInternalResult(data)
      onResult?.(data)
    } catch {
      addToast('Could not calculate perplexity', 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Perplexity Calculator</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5 space-y-3">
        <Textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Enter text to calculate perplexity..."
          rows={3}
        />
        <Button size="sm" onClick={handleCalculate} disabled={loading || !text.trim()} className="h-7 text-[11px]">
          {loading ? 'Calculating...' : 'Calculate'}
        </Button>
        {result && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5">
            <div className="rounded-lg bg-muted/20 p-2.5 text-center">
              <div className="text-[10px] text-muted-foreground">Perplexity</div>
              <div className="text-[11px] font-mono font-medium">{result.perplexity}</div>
            </div>
            <div className="rounded-lg bg-muted/20 p-2.5 text-center">
              <div className="text-[10px] text-muted-foreground">Loss</div>
              <div className="text-[11px] font-mono font-medium">{result.loss}</div>
            </div>
            <div className="rounded-lg bg-muted/20 p-2.5 text-center">
              <div className="text-[10px] text-muted-foreground">Tokens</div>
              <div className="text-[11px] font-mono font-medium">{result.tokens}</div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
