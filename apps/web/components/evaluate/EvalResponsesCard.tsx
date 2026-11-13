'use client'

import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'

interface LoggedResponse {
  user_message: string
  assistant_response: string
  model: string
  tokens_generated: number
  duration_ms?: number
  timestamp?: string
}

interface EvalResponsesCardProps {
  responses: LoggedResponse[]
  onRefresh?: () => void
  onClear?: () => void
}

export function EvalResponsesCard({ responses, onRefresh, onClear }: EvalResponsesCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Logged Responses ({responses.length})</CardTitle>
        <div className="flex gap-1">
          {onRefresh && (
            <Button size="sm" variant="ghost" onClick={onRefresh} aria-label="Refresh responses" className="h-6 text-[10px]">
              <IconRefresh className="h-4 w-4" />
            </Button>
          )}
          {onClear && (
            <Button size="sm" variant="ghost" className="text-destructive h-6 text-[10px]" onClick={onClear}>
              Clear
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5">
        {responses.length === 0 ? (
          <div className="text-center py-6 text-[10px] text-muted-foreground/60">
            No responses logged yet.
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {responses.map((r, i) => (
              <div key={i} className="rounded-lg border border-border/40 p-2.5 hover:bg-muted/20 text-sm">
                <div className="text-[10px] text-muted-foreground mb-1">
                  {r.timestamp ? new Date(r.timestamp).toLocaleString() : '—'} · {r.model} · {r.tokens_generated} tokens · {r.duration_ms?.toFixed(0)}ms
                </div>
                <div className="text-[10px]"><span className="text-muted-foreground">User:</span> {r.user_message}</div>
                <div className="text-[10px] mt-0.5"><span className="text-muted-foreground">AI:</span> {r.assistant_response}</div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
