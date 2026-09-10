import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { ArrowLeft, Database, Key } from 'lucide-react'

const RESOURCE_ICONS: Record<string, typeof Database> = {
  dataset: Database,
  knowledge: Key,
  api_key: Key,
}

const RESOURCE_COLORS: Record<string, string> = {
  dataset: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  knowledge: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
  api_key: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
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

interface IncomingSharesCardProps {
  shares: Share[]
  resolveName: (type: string, id: string) => string
  resolveWorkspace: (id: string) => string
}

export function IncomingSharesCard({ shares, resolveName, resolveWorkspace }: IncomingSharesCardProps) {
  return (
    <Card className="mb-4">
      <CardHeader className="pb-2">
        <CardTitle className="text-xs flex items-center gap-2">
          <ArrowLeft className="h-3 w-3" />
          Shared With Me ({shares.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">
        {shares.length === 0 ? (
          <p className="text-[10px] text-muted-foreground text-center py-4">No data shared with this workspace</p>
        ) : (
          shares.map(s => {
            const Icon = RESOURCE_ICONS[s.resource_type] || Database
            return (
              <div key={s.id} className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${RESOURCE_COLORS[s.resource_type] || 'bg-gray-100 text-gray-700'}`}>
                      <Icon className="h-2.5 w-2.5 inline mr-0.5" />
                      {s.resource_type}
                    </span>
                    <span className="font-medium">{resolveName(s.resource_type, s.resource_id)}</span>
                  </div>
                  <div className="text-muted-foreground mt-0.5 flex items-center gap-1">
                    From <span className="font-medium">{resolveWorkspace(s.source_workspace_id)}</span> · {s.permission}
                  </div>
                </div>
                <span className="text-muted-foreground">{new Date(s.shared_at).toLocaleDateString()}</span>
              </div>
            )
          })
        )}
      </CardContent>
    </Card>
  )
}
