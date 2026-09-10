'use client'

import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button, Input } from '@sloughgpt/strui'
import { IconPlus } from '@/components/icons/NavIcons'

interface ApiKeyCreateCardProps {
  newName: string
  onNameChange: (name: string) => void
  onCreate: () => void
  creating?: boolean
}

export function ApiKeyCreateCard({
  newName,
  onNameChange,
  onCreate,
  creating = false,
}: ApiKeyCreateCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Create API Key</CardTitle>
        <CardDescription>
          Create a new API key for programmatic access to the workspace.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Input
            value={newName}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder="Key name"
            maxLength={100}
            onKeyDown={(e) => e.key === 'Enter' && onCreate()}
          />
          <Button onClick={onCreate} disabled={!newName.trim() || creating}>
            <IconPlus className="mr-2 h-4 w-4" />
            {creating ? 'Creating...' : 'Create'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
