'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Textarea, Slider } from '@sloughgpt/strui'

interface KbAddEntryFormProps {
  content: string
  topic: string
  importance: number
  loading: boolean
  suggestResult: string | null
  onContentChange: (value: string) => void
  onTopicChange: (value: string) => void
  onImportanceChange: (value: number) => void
  onAdd: () => void
  onSuggest: () => void
}

export function KbAddEntryForm({
  content,
  topic,
  importance,
  loading,
  suggestResult,
  onContentChange,
  onTopicChange,
  onImportanceChange,
  onAdd,
  onSuggest,
}: KbAddEntryFormProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Add Knowledge Entry</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5 space-y-4">
        <div className="space-y-2">
          <Label className="text-xs">Content</Label>
          <Textarea
            value={content}
            onChange={e => onContentChange(e.target.value)}
            rows={4}
            className="text-xs"
            placeholder="Enter knowledge content..."
          />
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          <div className="space-y-2">
            <Label className="text-xs">Topic</Label>
            <Input value={topic} onChange={e => onTopicChange(e.target.value)} className="h-7 text-[11px]" />
            {suggestResult && <p className="text-xs text-muted-foreground">Suggested: {suggestResult}</p>}
          </div>
          <div className="space-y-2">
            <Slider
              id="importance-slider"
              label="Importance"
              value={[importance]}
              min={0}
              max={1}
              step={0.1}
              showValue
              formatValue={(v) => v.toFixed(1)}
              onValueChange={([v]) => onImportanceChange(v)}
              size="sm"
            />
          </div>
        </div>
        <div className="flex gap-2">
          <Button onClick={onAdd} disabled={loading || !content.trim()} className="flex-1 h-7 text-[11px]">
            {loading ? 'Adding...' : 'Add Entry'}
          </Button>
          <Button variant="outline" onClick={onSuggest} disabled={!content.trim()} className="h-7 text-[11px]">
            Suggest Topic
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
