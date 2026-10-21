import { cn } from '@sloughgpt/strui'
import type { AssetEvent } from '@/lib/files-controller'

/**
 * Asset event stream (put/download/delete/migrate) with action badges.
 *
 * Presentational only — the owning page fetches (filesController.history)
 * and owns refresh policy. Shared by the /files History tab and the live
 * /developer History tab so one implementation serves both surfaces.
 */
export function HistoryStream({ events, loading }: { events: AssetEvent[]; loading: boolean }) {
  if (loading) return <div className="h-32 animate-pulse bg-muted/20 rounded-lg" />
  if (events.length === 0) return <p className="text-[10px] text-muted-foreground/60">No asset events recorded yet.</p>
  return (
    <div className="space-y-0.5 max-h-96 overflow-y-auto" data-testid="history-stream">
      {events.map((e, i) => (
        <div key={`${e.ts}-${i}`} className="flex items-center gap-2 rounded-lg border border-border/40 px-2.5 py-1.5 text-[10px]">
          <span
            className={cn(
              'px-1.5 py-0.5 rounded-full font-medium shrink-0',
              e.action === 'delete' ? 'bg-destructive/10 text-destructive'
                : e.action === 'download' ? 'bg-success/10 text-success'
                : e.action === 'migrate' ? 'bg-muted/40 text-muted-foreground'
                : 'bg-primary/10 text-primary',
            )}
          >
            {e.action}
          </span>
          <span className="font-medium truncate flex-1">
            {e.original_name || (e.asset_id ? e.asset_id.slice(0, 8) : '—')}
          </span>
          {e.origin && <span className="text-muted-foreground/60 shrink-0">{e.origin}</span>}
          <span className="font-mono tabular-nums text-muted-foreground/60 shrink-0">
            {new Date(e.ts * 1000).toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  )
}
