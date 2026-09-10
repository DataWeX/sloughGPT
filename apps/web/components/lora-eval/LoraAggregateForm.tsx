'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Input } from '@sloughgpt/strui'

interface LoraAggregateFormProps {
  topK: number
  minFeedback: number
  aggregating: boolean
  onTopKChange: (value: number) => void
  onMinFeedbackChange: (value: number) => void
  onAggregate: () => void
}

export function LoraAggregateForm({
  topK,
  minFeedback,
  aggregating,
  onTopKChange,
  onMinFeedbackChange,
  onAggregate,
}: LoraAggregateFormProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Aggregate Adapters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-muted-foreground">Top K</label>
            <Input type="number" value={topK} onChange={e => onTopKChange(parseInt(e.target.value) || 10)} className="h-8 text-xs mt-1" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Min Feedback</label>
            <Input type="number" value={minFeedback} onChange={e => onMinFeedbackChange(parseInt(e.target.value) || 5)} className="h-8 text-xs mt-1" />
          </div>
        </div>
        <Button size="sm" onClick={onAggregate} disabled={aggregating}>
          {aggregating ? 'Aggregating...' : 'Aggregate'}
        </Button>
      </CardContent>
    </Card>
  )
}
