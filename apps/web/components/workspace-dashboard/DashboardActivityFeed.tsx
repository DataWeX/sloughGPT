'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { RefreshCw, ExternalLink } from 'lucide-react'

interface Activity {
  type: string
  action: string
  detail: string
  status: string
  timestamp: string
  user: string
}

interface DashboardActivityFeedProps {
  activities: Activity[]
  onRefresh?: () => void
  getActivityLink?: (activity: Activity) => string | null
  maxVisible?: number
}

function formatTime(ts: string) {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}

export function DashboardActivityFeed({
  activities,
  onRefresh,
  getActivityLink,
  maxVisible = 10,
}: DashboardActivityFeedProps) {
  const [showAll, setShowAll] = useState(false)
  const displayed = showAll ? activities : activities.slice(0, maxVisible)

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs">Recent Activity</CardTitle>
          {onRefresh && (
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onRefresh} title="Refresh">
              <RefreshCw className="h-3 w-3" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-1">
        {activities.length === 0 ? (
          <p className="text-xs text-muted-foreground py-4 text-center">No recent activity</p>
        ) : (
          <>
            {displayed.map((a, i) => {
              const link = getActivityLink?.(a)
              return (
                <div
                  key={i}
                  className={`flex items-center justify-between px-2 py-1.5 rounded text-[10px] ${
                    link ? 'hover:bg-muted/50 cursor-pointer' : ''
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    <span className="font-medium">{a.action}</span>
                    {a.detail && <span className="text-muted-foreground ml-1.5 truncate">{a.detail}</span>}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {a.user && <span className="text-muted-foreground">{a.user}</span>}
                    {a.status && (
                      <span
                        className={`px-1.5 py-0.5 rounded-full text-[9px] ${
                          a.status === 'completed' || a.status === 'success'
                            ? 'bg-green-100 text-green-700'
                            : a.status === 'failed'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-muted text-muted-foreground'
                        }`}
                      >
                        {a.status}
                      </span>
                    )}
                    <span className="text-muted-foreground whitespace-nowrap">{formatTime(a.timestamp)}</span>
                    {link && <ExternalLink className="h-2.5 w-2.5 text-muted-foreground" />}
                  </div>
                </div>
              )
            })}
            {activities.length > maxVisible && (
              <button
                onClick={() => setShowAll(!showAll)}
                className="w-full text-center text-[10px] text-muted-foreground hover:text-foreground py-2"
              >
                {showAll ? 'Show less' : `Show all ${activities.length} activities`}
              </button>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
