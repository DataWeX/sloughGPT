'use client'

import { Card, CardHeader, CardTitle, CardContent, cn } from '@sloughgpt/strui'

interface WorkspaceInfo {
  id: string
  name: string
  tenant_id: string
  description: string
  role: string
}

const ROLE_COLORS: Record<string, string> = {
  owner: 'bg-primary/15 text-primary',
  admin: 'bg-warning/15 text-warning',
  member: 'bg-success/15 text-success',
  viewer: 'bg-muted text-muted-foreground',
}

interface AuthWorkspaceCardProps {
  workspaces: WorkspaceInfo[]
  activeWorkspace?: string
  onSelect?: (id: string) => void
}

export function AuthWorkspaceCard({ workspaces, activeWorkspace, onSelect }: AuthWorkspaceCardProps) {
  if (workspaces.length === 0) return null

  return (
    <Card data-testid="auth-workspace">
      <CardHeader>
        <CardTitle className="text-base">
          Workspaces
          <span className="text-muted-foreground font-normal ml-2">({workspaces.length})</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1.5">
          {workspaces.map(ws => (
            <button
              key={ws.id}
              className={cn(
                'w-full text-left p-2.5 rounded-lg border transition-all',
                activeWorkspace === ws.id
                  ? 'border-primary/50 bg-primary/5'
                  : 'border-border hover:border-primary/30'
              )}
              onClick={() => onSelect?.(ws.id)}
              data-testid={`workspace-${ws.id}`}
            >
              <div className="flex items-center justify-between">
                <div className="text-xs font-medium">{ws.name}</div>
                <span className={cn(
                  'text-[9px] px-1.5 py-0.5 rounded font-medium',
                  ROLE_COLORS[ws.role] ?? 'bg-muted text-muted-foreground'
                )}>
                  {ws.role}
                </span>
              </div>
              {ws.description && (
                <div className="text-[10px] text-muted-foreground mt-0.5 line-clamp-1">{ws.description}</div>
              )}
              <div className="text-[9px] text-muted-foreground/60 mt-0.5 font-mono">{ws.tenant_id}</div>
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
