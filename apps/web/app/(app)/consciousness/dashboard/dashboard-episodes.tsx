'use client'

/**
 * Recent Episodes timeline and the episode detail dialog.
 *
 * The list stays dumb: it renders the episodes it is handed and reports the
 * two interactions (select, rate) upward. Rating-in-progress and the selected
 * episode are owned by `useConsciousnessDashboard`.
 */
import { Badge, Button, Card, CardContent, Dialog, DialogContent, DialogHeader, DialogOverlay, DialogPortal, DialogTitle, SectionHeader } from '@sloughgpt/strui'
import type { Episode } from '@/hooks/useConsciousnessDashboard'
import { QUALIA_COLORS, formatTimeAgo } from './dashboard-format'

interface DashboardEpisodesProps {
  episodes: Episode[]
  ratingEpisode: number | null
  onSelectEpisode: (episode: Episode) => void
  onRate: (episodeIndex: number, rating: number) => void
}

export function DashboardEpisodes({
  episodes,
  ratingEpisode,
  onSelectEpisode,
  onRate,
}: DashboardEpisodesProps) {
  return (
    <>
      {/* Timeline */}
      <SectionHeader
        title="Recent Episodes"
        description="Rate episodes to influence consciousness evolution"
        action={
          <Badge variant="outline" className="text-xs">
            {episodes.length} shown
          </Badge>
        }
      />
      <Card>
        <CardContent>
          {episodes.length > 0 ? (
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {[...episodes].reverse().map((ep, i) => {
                const realIndex = episodes.length - 1 - i
                return (
                  <div
                    key={i}
                    className="flex gap-2 sm:gap-3 rounded-lg border p-2 sm:p-3 text-xs sm:text-sm cursor-pointer hover:bg-muted/50 transition-colors"
                    onClick={() => onSelectEpisode(ep)}
                  >
                    <div className="flex flex-col items-center gap-1 text-[10px] sm:text-xs text-muted-foreground min-w-[48px] sm:min-w-[60px]">
                      <span>{formatTimeAgo(ep.timestamp)}</span>
                      <span
                        className={`font-mono ${ep.growth_delta >= 0 ? 'text-success' : 'text-destructive'}`}
                      >
                        {ep.growth_delta >= 0 ? '+' : ''}
                        {(ep.growth_delta * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[10px] sm:text-xs text-muted-foreground truncate">
                        {ep.input_text}
                      </div>
                      <div className="mt-1 text-xs sm:text-sm">{ep.self_insight}</div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {Object.entries(ep.qualia)
                          .slice(0, 4)
                          .map(([k, v]) => (
                            <span key={k} className="text-[10px] text-muted-foreground">
                              {k}: {v.toFixed(2)}
                            </span>
                          ))}
                      </div>
                      <div className="mt-2 flex gap-1">
                        {[1, 2, 3, 4, 5].map((r) => (
                          <Button
                            key={r}
                            size="sm"
                            variant={ratingEpisode === realIndex ? 'default' : 'outline'}
                            className="h-6 px-2 text-[10px]"
                            onClick={() => onRate(realIndex, r)}
                            disabled={ratingEpisode === realIndex}
                          >
                            {r}
                          </Button>
                        ))}
                        <span className="text-[10px] text-muted-foreground ml-1 self-center">
                          rate
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="flex h-24 sm:h-[100px] items-center justify-center text-sm text-muted-foreground">
              No episodes yet. Chat with consciousness enabled to start recording.
            </div>
          )}
        </CardContent>
      </Card>
    </>
  )
}

interface EpisodeDetailDialogProps {
  episode: Episode | null
  episodes: Episode[]
  onClose: () => void
  onRate: (episodeIndex: number, rating: number) => void
}

export function EpisodeDetailDialog({
  episode,
  episodes,
  onClose,
  onRate,
}: EpisodeDetailDialogProps) {
  return (
    <Dialog open={episode !== null} onOpenChange={(open) => !open && onClose()}>
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
                            width: `${Math.max(0, Math.min(100, ((v + 1) / 2) * 100))}%`,
                            backgroundColor: QUALIA_COLORS[k] || 'rgb(var(--chart-1))',
                          }}
                        />
                      </div>
                      <span className="text-[10px] text-muted-foreground w-8 text-right">
                        {v.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">Timestamp</div>
                <div className="text-xs text-muted-foreground">
                  {new Date(episode.timestamp * 1000).toLocaleString()}
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
