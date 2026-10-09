'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Checkbox, Skeleton, EmptyCard, cn } from '@sloughgpt/strui'
import { IconRefresh } from '@/components/icons/NavIcons'

interface Agent {
  id: string
  name: string
  description?: string
  tools: string[]
  instructions?: string
}

interface AgentListToolbarProps {
  agents: Agent[]
  loading: boolean
  selectedIds: Set<string>
  search: string
  onSearchChange: (value: string) => void
  onSelectAll: () => void
  onRefresh: () => void
  onBulkExport: () => void
  onBulkDelete: () => void
  renderAgent: (agent: Agent) => React.ReactNode
}

export function AgentListToolbar({
  agents,
  loading,
  selectedIds,
  search,
  onSearchChange,
  onSelectAll,
  onRefresh,
  onBulkExport,
  onBulkDelete,
  renderAgent,
}: AgentListToolbarProps) {
  const filtered = search
    ? agents.filter(a =>
        a.name.toLowerCase().includes(search.toLowerCase()) ||
        (a.description?.toLowerCase() ?? '').includes(search.toLowerCase()),
      )
    : agents

  return (
    <Card data-testid="agent-list-toolbar">
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <div className="flex items-center justify-between">
          <CardTitle className="text-[11px] font-medium">Agents</CardTitle>
          <Button size="sm" variant="ghost" onClick={onRefresh} disabled={loading} className="h-6 text-[10px]">
            <IconRefresh className="h-4 w-4 mr-1" />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5 space-y-2">
        {loading ? (
          <div className="space-y-2" data-testid="agents-loading">
            {[1, 2, 3].map(i => (
              <div key={i} className="flex items-center gap-1.5 p-2.5 rounded-lg border border-border/40">
                <Skeleton className="h-6 w-6 rounded shrink-0 bg-muted/20" />
                <div className="flex-1 space-y-1">
                  <Skeleton className="h-3.5 w-24 bg-muted/20 rounded-lg" />
                  <Skeleton className="h-3 w-40 bg-muted/20 rounded-lg" />
                </div>
                <Skeleton className="h-6 w-14 rounded-full bg-muted/20" />
              </div>
            ))}
          </div>
        ) : agents.length === 0 ? (
          <EmptyCard
            message="No agents yet"
            description="Create your first agent above. Use templates for quick setup."
            action={null}
          />
        ) : (
          <>
            {agents.length > 2 && (
              <input
                type="text"
                aria-label="Search agents"
                value={search}
                onChange={e => onSearchChange(e.target.value)}
                placeholder="Search agents..."
                className="h-7 w-full max-w-xs rounded-md border border-border/60 bg-background px-2.5 text-[11px] placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/30"
              />
            )}
            {filtered.length > 0 && (
              <div className="flex items-center gap-2 text-[10px]" data-testid="bulk-actions">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <Checkbox
                    checked={selectedIds.size === filtered.length && filtered.length > 0}
                    onCheckedChange={onSelectAll}
                    aria-label="Select all agents"
                    className="rounded border-border"
                  />
                  <span className="text-muted-foreground">Select all</span>
                </label>
                {selectedIds.size > 0 && (
                  <>
                    <span className="text-muted-foreground">·</span>
                    <span className="text-muted-foreground">{selectedIds.size} selected</span>
                    <Button size="sm" variant="ghost" onClick={onBulkExport} className="h-6 text-[10px]">
                      Export
                    </Button>
                    <Button size="sm" variant="ghost" onClick={onBulkDelete} className="h-6 text-[10px] text-destructive hover:text-destructive">
                      Delete
                    </Button>
                  </>
                )}
              </div>
            )}
            {filtered.length === 0 ? (
              <div className="text-center py-4">
                <p className="text-[10px] text-muted-foreground/60">No agents matching &quot;{search}&quot;</p>
                <Button size="sm" variant="ghost" className="h-6 text-[10px] mt-2" onClick={() => onSearchChange('')}>
                  Clear search
                </Button>
              </div>
            ) : (
              filtered.map(agent => renderAgent(agent))
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
