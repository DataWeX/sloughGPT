'use client'

import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'

interface ModelInfo {
  model_id: string
  model_type: string
  num_parameters: number
  vocab_size: number
  max_context: number
  num_layers: number
  has_tokenizer: boolean
  has_streaming: boolean
  has_embedding: boolean
}

interface InferModelInfoCardProps {
  info: ModelInfo | null
  loading: boolean
  onLoad: () => void
}

function StatBox({ label, value }: { label: string; value: string | number | boolean }) {
  const display = typeof value === 'boolean' ? (value ? 'Yes' : 'No') : value
  return (
    <div className="rounded-lg border border-border/50 bg-muted/30 p-2.5 space-y-0.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-medium font-mono truncate" title={String(display)}>
        {display}
      </div>
    </div>
  )
}

export function InferModelInfoCard({ info, loading, onLoad }: InferModelInfoCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Model Information</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Button onClick={onLoad} disabled={loading} className="w-full">
          {loading ? 'Loading...' : 'Load Info'}
        </Button>

        {info && (
          <div className="grid grid-cols-3 gap-2">
            <StatBox label="Model ID" value={info.model_id} />
            <StatBox label="Type" value={info.model_type} />
            <StatBox label="Parameters" value={info.num_parameters.toLocaleString()} />
            <StatBox label="Vocab Size" value={info.vocab_size.toLocaleString()} />
            <StatBox label="Max Context" value={info.max_context.toLocaleString()} />
            <StatBox label="Layers" value={info.num_layers} />
            <StatBox label="Tokenizer" value={info.has_tokenizer} />
            <StatBox label="Streaming" value={info.has_streaming} />
            <StatBox label="Embedding" value={info.has_embedding} />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
