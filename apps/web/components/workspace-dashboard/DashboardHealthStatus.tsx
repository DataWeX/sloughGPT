'use client'

import { Card, CardContent, Button } from '@sloughgpt/strui'

interface DashboardHealthStatusProps {
  status: string | null
  checking?: boolean
  onRecheck?: () => void
}

export function DashboardHealthStatus({
  status,
  checking = false,
  onRecheck,
}: DashboardHealthStatusProps) {
  if (!status) return null

  const colorMap: Record<string, string> = {
    healthy: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
  }

  return (
    <Card className="mb-4">
      <CardContent className="py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs">
            <span className={`h-2 w-2 rounded-full ${colorMap[status] ?? 'bg-gray-500'}`} />
            <span className="font-medium capitalize">Workspace {status}</span>
          </div>
          {onRecheck && (
            <button
              onClick={onRecheck}
              disabled={checking}
              className="text-[10px] text-muted-foreground hover:text-foreground"
            >
              {checking ? 'Checking...' : 'Re-check'}
            </button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
