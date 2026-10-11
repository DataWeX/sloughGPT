/**
 * Episode detail modal — full breakdown of one episode with inline rating.
 *
 * Always mounted; `episode === null` keeps it closed, matching the previous
 * behaviour where the Dialog was rendered unconditionally.
 */

import {
  Button,
  Dialog,
  DialogContent,
  DialogHeader,
  DialogPortal,
  DialogOverlay,
  DialogTitle,
} from '@sloughgpt/strui'
import { formatDateTime, toDateSeconds } from '@/lib/time-format'
import { QUALIA_COLORS, type Episode } from './types'

export interface EpisodeDetailModalProps {
  episode: Episode | null
  /** Full episode list, used to resolve the rated episode's index. */
  episodes: Episode[]
  onRate: (episodeIndex: number, rating: number) => void
  onClose: () => void
}

export function EpisodeDetailModal({
  episode,
  episodes,
  onRate,
  onClose,
}: EpisodeDetailModalProps) {
  return (
    <Dialog
      open={episode !== null}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
    >
      <DialogPortal>
        <DialogOverlay />
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Episode Detail</DialogTitle>
          </DialogHeader>
          {episode && (
            <div className="space-y-4">
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">Input</div>
                <div className="text-sm rounded-md bg-muted p-2">{episode.input_text}</div>
              </div>
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">Self Insight</div>
                <div className="text-sm">{episode.self_insight}</div>
              </div>
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">Growth Delta</div>
                <div
                  className={`text-sm font-mono ${episode.growth_delta >= 0 ? 'text-success' : 'text-destructive'}`}
                >
                  {episode.growth_delta >= 0 ? '+' : ''}
                  {(episode.growth_delta * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2">
                  Qualia Breakdown
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(episode.qualia).map(([k, v]) => (
                    <div key={k} className="flex items-center gap-2">
                      <span className="text-xs capitalize w-20">{k}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(0, Math.min(100, (((v as number) + 1) / 2) * 100))}%`,
                            backgroundColor: QUALIA_COLORS[k] || 'var(--chart-1)',
                          }}
                        />
                      </div>
                      <span className="text-[10px] text-muted-foreground w-8 text-right">
                        {(v as number).toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">Timestamp</div>
                <div className="text-xs text-muted-foreground">
                  {formatDateTime(toDateSeconds(episode.timestamp))}
                </div>
              </div>
              <div className="flex gap-2 pt-2">
                {[1, 2, 3, 4, 5].map((r) => (
                  <Button
                    key={r}
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      const idx = episodes.indexOf(episode)
                      if (idx >= 0) onRate(idx, r)
                      onClose()
                    }}
                  >
                    {r}
                  </Button>
                ))}
                <span className="text-xs text-muted-foreground ml-1 self-center">
                  rate this episode
                </span>
              </div>
            </div>
          )}
        </DialogContent>
      </DialogPortal>
    </Dialog>
  )
}
