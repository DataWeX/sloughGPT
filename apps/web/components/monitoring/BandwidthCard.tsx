'use client'

import { memo } from 'react'
import { cn, Card, CardContent, Progress, StatCard, KpiGrid } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import { formatBytes } from '@/lib/format-bytes'

interface BandwidthCardProps {
  liveHealth: LiveHealthSnapshot | null
}

/**
 * Edge bandwidth — what compression saved on the wire.
 *
 * Reads the mirrored gateway counters (`bandwidth` block): identity bytes
 * are what clients would have received uncompressed, wire bytes are what
 * actually left the process. Hidden entirely when no gateway has ever been
 * reached (Python-only mode), so absence never masquerades as "0 saved".
 */
export const BandwidthCard = memo(function BandwidthCard({ liveHealth }: BandwidthCardProps) {
  const bw = liveHealth?.bandwidth
  if (!bw || bw.identity_bytes <= 0) return null

  const savedPct = bw.saved_pct
  const expanding = savedPct < 0
  // Share of identity mass that actually crossed the wire (0–100).
  const wireShare =
    bw.identity_bytes > 0
      ? Math.min(100, Math.max(0, (bw.wire_bytes / bw.identity_bytes) * 100))
      : 100
  const totalResponses = bw.compressed_responses + bw.identity_responses

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">
        Edge bandwidth
      </span>
      <CardContent className="p-0 space-y-3">
        <KpiGrid columns={3}>
          <StatCard
            label="Identity"
            value={formatBytes(bw.identity_bytes)}
            numeric
            icon={<span className="inline-block w-2 h-2 rounded-full bg-muted-foreground/50" />}
          />
          <StatCard
            label="On the wire"
            value={formatBytes(bw.wire_bytes)}
            numeric
            icon={<span className="inline-block w-2 h-2 rounded-full bg-primary" />}
          />
          <StatCard
            label="Saved"
            value={`${savedPct.toFixed(1)}%`}
            numeric
            icon={
              <span
                className={cn(
                  'inline-block w-2 h-2 rounded-full',
                  expanding ? 'bg-warning' : 'bg-success',
                )}
              />
            }
          />
        </KpiGrid>

        {/* Wire mass: how much of the identity payload survived the trip. */}
        <div className="space-y-1">
          <Progress
            value={wireShare}
            variant={expanding ? 'warning' : 'success'}
            size="sm"
            label="Wire share of identity bytes"
          />
          <p className="text-[10px] text-muted-foreground">
            {bw.compressed_responses} compressed / {totalResponses} responses
            {bw.zstd_responses > 0 && ` · ${bw.zstd_responses} zstd`}
            {bw.gzip_responses > 0 && ` · ${bw.gzip_responses} gzip`}
          </p>
        </div>
      </CardContent>
    </Card>
  )
})
