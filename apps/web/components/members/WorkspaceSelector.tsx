'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface Workspace {
  id: string
  name: string
  member_count: number
}

interface WorkspaceSelectorProps {
  workspaces: Workspace[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export function WorkspaceSelector({ workspaces, selectedId, onSelect }: WorkspaceSelectorProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Select Workspace</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          {workspaces.map(ws => (
            <button
              key={ws.id}
              onClick={() => onSelect(ws.id)}
              className={`px-3 py-1.5 rounded-md text-xs transition-colors ${
                selectedId === ws.id
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80 text-muted-foreground'
              }`}
            >
              {ws.name}
              <span className="ml-1.5 text-[10px] opacity-70">({ws.member_count})</span>
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
