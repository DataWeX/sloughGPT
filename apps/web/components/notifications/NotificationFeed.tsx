import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

export interface NotificationItem {
  type: string
  title: string
  detail: string
  status: string
  timestamp: string
}

export interface NotificationFeedProps {
  notifications: NotificationItem[]
}

const TYPE_STYLES: Record<string, string> = {
  training: 'bg-info/15 text-info dark:bg-info/10 dark:text-info',
  member: 'bg-primary/15 text-primary dark:bg-primary/10 dark:text-primary',
}

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-success/15 text-success dark:bg-success/10 dark:text-success',
  failed: 'bg-destructive/15 text-destructive dark:bg-destructive/10 dark:text-destructive',
}

function formatTime(ts: string) {
  if (!ts) return ''
  try { return new Date(ts).toLocaleString() } catch { return ts }
}

export function NotificationFeed({ notifications }: NotificationFeedProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Recent Events</CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">
        {notifications.length === 0 ? (
          <p className="text-xs text-muted-foreground py-4 text-center">No notifications</p>
        ) : (
          notifications.map((n, i) => (
            <div key={i} className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${TYPE_STYLES[n.type] || 'bg-muted text-muted-foreground'}`}>
                    {n.type}
                  </span>
                  <span className="font-medium">{n.title}</span>
                </div>
                {n.detail && <div className="text-muted-foreground mt-0.5">{n.detail}</div>}
              </div>
              <div className="flex items-center gap-3 shrink-0">
                {n.status && (
                  <span className={`px-1.5 py-0.5 rounded-full text-[9px] ${STATUS_STYLES[n.status] || 'bg-muted text-muted-foreground'}`}>
                    {n.status}
                  </span>
                )}
                <span className="text-muted-foreground whitespace-nowrap">{formatTime(n.timestamp)}</span>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
