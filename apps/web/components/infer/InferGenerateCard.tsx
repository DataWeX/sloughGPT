'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Textarea, Input, Label, cn } from '@sloughgpt/strui'

interface GenerateResult {
  text: string
  tokens_generated: number
  elapsed_ms: number
  model: string
}

interface InferGenerateCardProps {
  prompt: string
  maxTokens: number
  temperature: number
  topP: number
  topK: number
  repPenalty: number
  result: GenerateResult | null
  loading: boolean
  onPromptChange: (v: string) => void
  onMaxTokensChange: (v: number) => void
  onTemperatureChange: (v: number) => void
  onTopPChange: (v: number) => void
  onTopKChange: (v: number) => void
  onRepPenaltyChange: (v: number) => void
  onRun: () => void
}

export function InferGenerateCard({
  prompt,
  maxTokens,
  temperature,
  topP,
  topK,
  repPenalty,
  result,
  loading,
  onPromptChange,
  onMaxTokensChange,
  onTemperatureChange,
  onTopPChange,
  onTopKChange,
  onRepPenaltyChange,
  onRun,
}: InferGenerateCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Text Generation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <Label htmlFor="infer-prompt">Prompt</Label>
          <Textarea
            id="infer-prompt"
            value={prompt}
            onChange={e => onPromptChange(e.target.value)}
            placeholder="Enter your prompt..."
            className="h-24 text-sm resize-none"
            aria-label="Prompt"
          />
        </div>

        <div className="grid grid-cols-5 gap-2">
          <div className="space-y-1">
            <Label htmlFor="infer-max-tokens" className="text-xs">Max Tokens</Label>
            <Input
              id="infer-max-tokens"
              type="number"
              value={maxTokens}
              onChange={e => onMaxTokensChange(Number(e.target.value))}
              className="h-8 text-xs"
              aria-label="Max Tokens"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="infer-temperature" className="text-xs">Temperature</Label>
            <Input
              id="infer-temperature"
              type="number"
              step="0.1"
              value={temperature}
              onChange={e => onTemperatureChange(Number(e.target.value))}
              className="h-8 text-xs"
              aria-label="Temperature"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="infer-top-p" className="text-xs">Top P</Label>
            <Input
              id="infer-top-p"
              type="number"
              step="0.1"
              value={topP}
              onChange={e => onTopPChange(Number(e.target.value))}
              className="h-8 text-xs"
              aria-label="Top P"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="infer-top-k" className="text-xs">Top K</Label>
            <Input
              id="infer-top-k"
              type="number"
              value={topK}
              onChange={e => onTopKChange(Number(e.target.value))}
              className="h-8 text-xs"
              aria-label="Top K"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="infer-rep-penalty" className="text-xs">Rep Penalty</Label>
            <Input
              id="infer-rep-penalty"
              type="number"
              step="0.1"
              value={repPenalty}
              onChange={e => onRepPenaltyChange(Number(e.target.value))}
              className="h-8 text-xs"
              aria-label="Rep Penalty"
            />
          </div>
        </div>

        <Button onClick={onRun} disabled={loading || !prompt.trim()} className="w-full">
          {loading ? 'Generating...' : 'Run'}
        </Button>

        {result && (
          <div className="space-y-2">
            <pre className="rounded-lg border border-border/50 bg-muted/30 p-3 text-xs font-mono overflow-auto max-h-64 whitespace-pre-wrap">
              {result.text}
            </pre>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>{result.tokens_generated} tokens</span>
              <span>{result.elapsed_ms}ms</span>
              <span>{result.model}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
