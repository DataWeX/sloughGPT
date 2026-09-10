import { Card, CardContent, CardHeader, CardTitle, Button, Input } from '@sloughgpt/strui'

interface Dataset {
  id: string
  name: string
}

interface Workspace {
  id: string
  name: string
}

interface ShareDialogProps {
  shareType: string
  shareResourceId: string
  shareTargetWs: string
  sharePermission: string
  datasets: Dataset[]
  workspaces: Workspace[]
  onShareTypeChange: (value: string) => void
  onResourceIdChange: (value: string) => void
  onTargetWsChange: (value: string) => void
  onPermissionChange: (value: string) => void
  onShare: () => void
  onCancel: () => void
}

export function ShareDialog({
  shareType,
  shareResourceId,
  shareTargetWs,
  sharePermission,
  datasets,
  workspaces,
  onShareTypeChange,
  onResourceIdChange,
  onTargetWsChange,
  onPermissionChange,
  onShare,
  onCancel,
}: ShareDialogProps) {
  return (
    <Card className="mb-4 border-primary/30">
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Share Data with Another Workspace</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex gap-2">
          <select
            value={shareType}
            onChange={e => onShareTypeChange(e.target.value)}
            className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
          >
            <option value="dataset">Dataset</option>
            <option value="knowledge">Knowledge</option>
            <option value="api_key">API Key</option>
          </select>
          {shareType === 'dataset' ? (
            <select
              value={shareResourceId}
              onChange={e => onResourceIdChange(e.target.value)}
              className="flex-1 h-6 text-[10px] rounded-md border border-border bg-background px-2"
            >
              <option value="">Select dataset...</option>
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          ) : (
            <Input
              value={shareResourceId}
              onChange={e => onResourceIdChange(e.target.value)}
              placeholder={`${shareType} ID`}
              className="flex-1 h-6 text-[10px]"
            />
          )}
        </div>
        <div className="flex gap-2">
          <select
            value={shareTargetWs}
            onChange={e => onTargetWsChange(e.target.value)}
            className="flex-1 h-6 text-[10px] rounded-md border border-border bg-background px-2"
          >
            <option value="">Select workspace...</option>
            {workspaces.map(w => (
              <option key={w.id} value={w.id}>{w.name}</option>
            ))}
          </select>
          <select
            value={sharePermission}
            onChange={e => onPermissionChange(e.target.value)}
            className="h-6 text-[10px] rounded-md border border-border bg-background px-2"
          >
            <option value="read">Read only</option>
            <option value="admin">Admin</option>
          </select>
        </div>
        <div className="flex gap-2">
          <Button size="sm" className="h-6 text-[10px]" onClick={onShare}>Share</Button>
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={onCancel}>Cancel</Button>
        </div>
      </CardContent>
    </Card>
  )
}
