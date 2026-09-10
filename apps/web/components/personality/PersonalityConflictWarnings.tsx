import { Card, CardHeader, CardTitle, CardDescription, CardContent, Badge } from '@sloughgpt/strui'

export interface PersonalityConflict {
  type: string
  severity: string
  message: string
  fields: string[]
}

export interface PersonalityConflictWarningsProps {
  conflicts: PersonalityConflict[]
}

export function PersonalityConflictWarnings({ conflicts }: PersonalityConflictWarningsProps) {
  if (conflicts.length === 0) return null

  return (
    <Card className="border-yellow-500/50">
      <CardHeader>
        <CardTitle className="text-yellow-600 dark:text-yellow-400">Personality Conflicts</CardTitle>
        <CardDescription>These settings may work against each other</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {conflicts.map((c, i) => (
          <div key={i} className="flex items-start gap-2 text-sm p-2 rounded-md bg-yellow-500/5">
            <Badge variant={c.severity === 'medium' ? 'destructive' : 'outline'} className="text-[10px] mt-0.5">
              {c.severity}
            </Badge>
            <div>
              <div>{c.message}</div>
              <div className="text-[10px] text-muted-foreground">{c.fields.join(' + ')}</div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
