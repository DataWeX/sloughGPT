import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { formatLocaleDate } from '@/lib/time-format'
import { ArrowRight, Database, Key } from 'lucide-react'

const RESOURCE_ICONS: Record<string, typeof Database> = {
  dataset: Database,
  knowledge: Key,
  api_key: Key,
}

const RESOURCE_COLORS: Record<string, string> = {
  dataset: 'bg-success/15 text-success dark:bg-success/10 dark:text-success',
  knowledge: 'bg-accent text-accent dark:bg-accent/30 dark:text-accent',
  api_key: 'bg-primary/15 text-primary dark:bg-primary/10 dark:text-primary',
}

interface Share {
  id: string
  resource_type: string
  resource_id: string
  source_workspace_id: string
  target_workspace_id: string
  permission: string
  shared_by: string
  shared_at: string
}

interface OutgoingSharesCardProps {
  shares: Share[]
  resolveName: (type: string, id: string) => string
  resolveWorkspace: (id: string) => string
  onRevoke: (shareId: string) => void
}

export function OutgoingSharesCard({
  shares,
  resolveName,
  resolveWorkspace,
  onRevoke,
}: OutgoingSharesCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs flex items-center gap-2">
          <ArrowRight className="h-3 w-3" />
          Shared by Me ({shares.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">
        {shares.length === 0 ? (
          <p className="text-[10px] text-muted-foreground text-center py-4">
            No data shared from this workspace
          </p>
        ) : (
          shares.map((s) => {
            const Icon = RESOURCE_ICONS[s.resource_type] || Database
            return (
              <div
                key={s.id}
                className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${RESOURCE_COLORS[s.resource_type] || 'bg-muted/50 text-muted-foreground'}`}
                    >
                      <Icon className="h-2.5 w-2.5 inline mr-0.5" />
                      {s.resource_type}
                    </span>
                    <span className="font-medium">
                      {resolveName(s.resource_type, s.resource_id)}
                    </span>
                  </div>
                  <div className="text-muted-foreground mt-0.5 flex items-center gap-1">
                    To{' '}
                    <span className="font-medium">{resolveWorkspace(s.target_workspace_id)}</span> ·{' '}
                    {s.permission}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">{formatLocaleDate(s.shared_at)}</span>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-5 text-[9px] text-destructive"
                    onClick={() => onRevoke(s.id)}
                  >
                    Revoke
                  </Button>
                </div>
              </div>
            )
          })
        )}
      </CardContent>
    </Card>
  )
}
