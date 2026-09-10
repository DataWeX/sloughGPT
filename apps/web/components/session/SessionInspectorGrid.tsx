import { Card, CardContent, CardHeader, CardTitle, cn } from '@sloughgpt/strui'

interface SessionStats {
  messages?: number
  knowledgeFacts?: number
  feedback?: number
  elapsedMs?: number
  episodicCount?: number
  sensoryBufferSize?: number
  workingMemory?: string[]
  semanticKeys?: string[]
  systemPrompt?: string
}

interface SessionInspectorGridProps {
  stats?: SessionStats
  modes?: Record<string, string>
  traits?: Record<string, unknown>
}

export function SessionInspectorGrid({ stats, modes = {}, traits = {} }: SessionInspectorGridProps) {
  if (!stats) return null

  return (
    <>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {[
          { label: 'Messages', value: stats.messages },
          { label: 'Knowledge Facts', value: stats.knowledgeFacts },
          { label: 'Feedback', value: stats.feedback },
          { label: 'Inspect Time', value: stats.elapsedMs !== undefined ? `${stats.elapsedMs}ms` : undefined },
        ].map(s => (
          <div key={s.label} className="rounded-lg bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground/60">{s.label}</div>
            <div className="text-[13px] font-mono font-medium tabular-nums">{s.value ?? '--'}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Workspace</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="grid grid-cols-2 gap-1.5">
              <div className="rounded bg-muted/30 p-1.5 text-center">
                <div className="text-[10px] text-muted-foreground/60">Episodic Memory</div>
                <div className="text-[11px] font-mono tabular-nums">{stats.episodicCount ?? 0}</div>
              </div>
              <div className="rounded bg-muted/30 p-1.5 text-center">
                <div className="text-[10px] text-muted-foreground/60">Sensory Buffer</div>
                <div className="text-[11px] font-mono tabular-nums">{stats.sensoryBufferSize ?? 0}</div>
              </div>
            </div>
            {stats.workingMemory && stats.workingMemory.length > 0 && (
              <div>
                <p className="text-[10px] text-muted-foreground/60 mb-0.5">Working Memory</p>
                <div className="flex flex-wrap gap-0.5">
                  {stats.workingMemory.map((m, i) => (
                    <span key={i} className="rounded bg-primary/10 px-1 py-px text-[10px] text-primary">{m}</span>
                  ))}
                </div>
              </div>
            )}
            {stats.semanticKeys && stats.semanticKeys.length > 0 && (
              <div>
                <p className="text-[10px] text-muted-foreground/60 mb-0.5">Semantic Keys</p>
                <div className="flex flex-wrap gap-0.5">
                  {stats.semanticKeys.map((k, i) => (
                    <span key={i} className="rounded bg-muted/50 px-1 py-px text-[10px]">{k}</span>
                  ))}
                </div>
              </div>
            )}
            {stats.systemPrompt && (
              <div>
                <p className="text-[10px] text-muted-foreground/60 mb-0.5">System Prompt (truncated)</p>
                <pre className="max-h-[80px] overflow-y-auto rounded bg-muted/30 p-1.5 text-[10px] whitespace-pre-wrap">{stats.systemPrompt}</pre>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">Modes & Traits</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {Object.keys(modes).length > 0 && (
              <div className="grid grid-cols-2 gap-1.5">
                {Object.entries(modes).map(([k, v]) => (
                  <div key={k} className="rounded bg-muted/30 p-1.5">
                    <div className="text-[10px] text-muted-foreground/60 capitalize">{k}</div>
                    <div className="text-[11px] font-medium">{v}</div>
                  </div>
                ))}
              </div>
            )}
            {Object.keys(traits).length > 0 && (
              <div>
                <p className="text-[10px] text-muted-foreground/60 mb-0.5">Traits</p>
                <pre className="max-h-[120px] overflow-y-auto rounded bg-muted/30 p-1.5 text-[10px]">{JSON.stringify(traits, null, 2)}</pre>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  )
}
