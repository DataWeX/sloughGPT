'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface WorkspaceCreateCardProps {
  name: string
  description: string
  onNameChange: (name: string) => void
  onDescriptionChange: (desc: string) => void
  onCreate: () => void
  creating?: boolean
}

export function WorkspaceCreateCard({
  name,
  description,
  onNameChange,
  onDescriptionChange,
  onCreate,
  creating = false,
}: WorkspaceCreateCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Create Workspace</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex gap-1.5">
          <Input
            value={name}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder="Workspace name"
            className="flex-1 h-6 text-[10px]"
          />
          <Input
            value={description}
            onChange={(e) => onDescriptionChange(e.target.value)}
            placeholder="Description (optional)"
            className="flex-1 h-6 text-[10px]"
          />
          <Button
            size="sm"
            className="h-6 text-[10px]"
            onClick={onCreate}
            disabled={creating || !name.trim()}
          >
            {creating ? 'Creating...' : 'Create'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
