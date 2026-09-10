'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface WorkspaceGeneralSettingsCardProps {
  name: string
  description: string
  onNameChange: (name: string) => void
  onDescriptionChange: (desc: string) => void
  hasChanges?: boolean
  saving?: boolean
  onSave?: () => void
}

export function WorkspaceGeneralSettingsCard({
  name,
  description,
  onNameChange,
  onDescriptionChange,
  hasChanges = false,
  saving = false,
  onSave,
}: WorkspaceGeneralSettingsCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>General</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Name</label>
          <Input
            value={name}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder="Workspace name"
          />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Description</label>
          <textarea
            value={description}
            onChange={(e) => onDescriptionChange(e.target.value)}
            placeholder="Describe this workspace"
            rows={3}
            className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        {onSave && (
          <div className="flex items-center justify-between pt-2">
            <span className="text-sm text-muted-foreground">
              {hasChanges ? 'Unsaved changes' : 'No changes'}
            </span>
            <Button onClick={onSave} disabled={!hasChanges || saving}>
              {saving ? 'Saving...' : 'Save'}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
