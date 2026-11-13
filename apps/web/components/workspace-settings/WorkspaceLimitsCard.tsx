'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface WorkspaceLimitsCardProps {
  defaultModel: string
  dataRetentionDays: number
  maxMembers: number
  allowSharing: boolean
  onDefaultModelChange: (model: string) => void
  onDataRetentionChange: (days: number) => void
  onMaxMembersChange: (max: number) => void
  onAllowSharingChange: (allow: boolean) => void
  hasChanges?: boolean
  saving?: boolean
  onSave?: () => void
}

export function WorkspaceLimitsCard({
  defaultModel,
  dataRetentionDays,
  maxMembers,
  allowSharing,
  onDefaultModelChange,
  onDataRetentionChange,
  onMaxMembersChange,
  onAllowSharingChange,
  hasChanges = false,
  saving = false,
  onSave,
}: WorkspaceLimitsCardProps) {
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Defaults</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Default Model</label>
            <Input
              value={defaultModel}
              onChange={(e) => onDefaultModelChange(e.target.value)}
              placeholder="e.g. llama-3.2-3b"
            />
            <p className="text-xs text-muted-foreground">
              Model used by default for new conversations in this workspace
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Limits</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Data Retention (days)</label>
              <Input
                type="number"
                min={7}
                max={365}
                value={dataRetentionDays}
                onChange={(e) => onDataRetentionChange(parseInt(e.target.value) || 90)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Max Members</label>
              <Input
                type="number"
                min={2}
                max={500}
                value={maxMembers}
                onChange={(e) => onMaxMembersChange(parseInt(e.target.value) || 50)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Sharing</CardTitle>
        </CardHeader>
        <CardContent>
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={allowSharing}
              onChange={(e) => onAllowSharingChange(e.target.checked)}
              className="h-4 w-4 rounded border-border"
            />
            <div>
              <span className="text-sm font-medium text-foreground">Allow data sharing</span>
              <p className="text-xs text-muted-foreground">
                Allow members to share datasets and knowledge with other workspaces
              </p>
            </div>
          </label>
        </CardContent>
      </Card>

      {onSave && (
        <div className="flex items-center justify-between rounded-lg border border-border bg-card p-4">
          <div className="text-sm text-muted-foreground">
            {hasChanges ? 'You have unsaved changes' : 'No changes'}
          </div>
          <Button onClick={onSave} disabled={!hasChanges || saving}>
            {saving ? 'Saving...' : 'Save Settings'}
          </Button>
        </div>
      )}
    </>
  )
}
