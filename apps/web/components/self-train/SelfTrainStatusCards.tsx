import { Card, CardContent } from '@sloughgpt/strui'

interface SelfTrainStatusCardsProps {
  status?: 'not_started' | 'running' | 'exited' | null
  pid?: number
  returncode?: number
  historyCount?: number
  loading?: boolean
}

const statusLabel = (status?: 'not_started' | 'running' | 'exited' | null, loading?: boolean) => {
  if (loading) return '...'
  if (status === 'running') return 'Running'
  if (status === 'exited') return 'Exited'
  return 'Not started'
}

export function SelfTrainStatusCards({ status, pid, returncode, historyCount = 0, loading }: SelfTrainStatusCardsProps) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Card>
        <CardContent className="p-3">
          <p className="text-xs text-muted-foreground">Status</p>
          <p className="text-base font-medium">{statusLabel(status, loading)}</p>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-3">
          <p className="text-xs text-muted-foreground">PID</p>
          <p className="text-base font-medium font-mono">{pid ?? '--'}</p>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-3">
          <p className="text-xs text-muted-foreground">Exit code</p>
          <p className="text-base font-medium">{returncode ?? '--'}</p>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-3">
          <p className="text-xs text-muted-foreground">History lines</p>
          <p className="text-base font-medium">{historyCount}</p>
        </CardContent>
      </Card>
    </div>
  )
}
