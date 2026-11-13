'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input, Label } from '@sloughgpt/strui'

export interface TokenTreeTrainCardProps {
  vocabSize: number
  trainTexts: string
  loading: boolean
  onVocabSizeChange: (size: number) => void
  onTrainTextsChange: (text: string) => void
  onTrain: () => void
}

export function TokenTreeTrainCard({
  vocabSize,
  trainTexts,
  loading,
  onVocabSizeChange,
  onTrainTextsChange,
  onTrain,
}: TokenTreeTrainCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Train Tree</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 px-2.5 pb-2.5">
        <div className="grid grid-cols-2 gap-2">
          <div className="space-y-0.5">
            <Label className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Vocab Size</Label>
            <Input
              type="number"
              value={vocabSize}
              onChange={e => onVocabSizeChange(Number(e.target.value))}
              className="h-7 text-[11px] font-mono"
            />
          </div>
          <div className="space-y-0.5">
            <Label className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">Training Texts (one per line, optional)</Label>
            <textarea
              value={trainTexts}
              onChange={e => onTrainTextsChange(e.target.value)}
              rows={3}
              className="w-full rounded-lg border border-border/40 bg-background px-2.5 py-2 text-[11px] font-mono"
              placeholder="Leave empty to use default corpus..."
            />
          </div>
        </div>
        <Button onClick={onTrain} disabled={loading} className="w-full h-7 text-[11px]">
          {loading ? 'Training...' : 'Train Token Tree'}
        </Button>
      </CardContent>
    </Card>
  )
}
