import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label } from '@sloughgpt/strui'

interface SessionLoadCardProps {
  sessionId: string
  loading?: boolean
  regenerating?: boolean
  onSessionIdChange: (value: string) => void
  onInspect: () => void
  onRegenerate: () => void
}

export function SessionLoadCard({
  sessionId,
  loading,
  regenerating,
  onSessionIdChange,
  onInspect,
  onRegenerate,
}: SessionLoadCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Load Session</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="space-y-1">
          <Label className="text-xs">Session ID</Label>
          <Input
            data-testid="input"
            value={sessionId}
            onChange={(e) => onSessionIdChange(e.target.value)}
            placeholder="Enter session ID..."
            className="h-6 text-xs font-mono"
            onKeyDown={(e) => e.key === 'Enter' && onInspect()}
          />
        </div>
        <div className="flex gap-1.5">
          <Button
            onClick={onInspect}
            disabled={loading || !sessionId.trim()}
            className="flex-1 h-6 text-xs"
          >
            {loading ? 'Loading...' : 'Inspect'}
          </Button>
          <Button
            onClick={onRegenerate}
            disabled={regenerating || !sessionId.trim()}
            variant="outline"
            className="flex-1 h-6 text-xs"
          >
            {regenerating ? 'Regenerating...' : 'Regenerate Last Response'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
