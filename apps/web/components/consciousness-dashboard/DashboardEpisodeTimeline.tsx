/**
 * Recent episodes timeline — newest first, each row inline-rateable and
 * clickable to open the detail modal.
 */

import { Badge, Button, Card, CardContent, SectionHeader } from '@sloughgpt/strui'
import { formatTimeAgo, type Episode } from './types'

export interface DashboardEpisodeTimelineProps {
  episodes: Episode[]
  ratingEpisode: number | null
  onFeedback: (episodeIndex: number, rating: number) => void
  onSelect: (episode: Episode) => void
}

export function DashboardEpisodeTimeline({
  episodes,
  ratingEpisode,
  onFeedback,
  onSelect,
}: DashboardEpisodeTimelineProps) {
  return (
    <>
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
                    onClick={() => onSelect(ep)}
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
                              {k}: {(v as number).toFixed(2)}
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
                            onClick={() => onFeedback(realIndex, r)}
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
