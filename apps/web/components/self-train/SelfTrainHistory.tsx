import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'

interface SelfTrainHistoryProps {
  history?: string[]
}

export function SelfTrainHistory({ history }: SelfTrainHistoryProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Training history</CardTitle>
      </CardHeader>
      <CardContent>
        {(!history || history.length === 0) ? (
          <p className="text-xs text-muted-foreground">No history yet.</p>
        ) : (
          <div className="max-h-[400px] overflow-y-auto rounded bg-muted/30 p-3 font-mono text-xs">
            {history.map((line, i) => (
              <div key={i} className="whitespace-pre-wrap">{line}</div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
