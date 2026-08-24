'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

interface GalleryImage {
  id: string
  path: string
  created: number
}

interface ImageGalleryInsightsCardProps {
  gallery: GalleryImage[]
  styles: Array<{ key: string; name: string }>
}

function computeStats(gallery: GalleryImage[]) {
  if (gallery.length === 0) return null

  const now = Date.now() / 1000
  const last24h = gallery.filter(i => now - i.created < 86400).length
  const last7d = gallery.filter(i => now - i.created < 604800).length
  const oldest = Math.min(...gallery.map(i => i.created))
  const newest = Math.max(...gallery.map(i => i.created))
  const avgGap = gallery.length > 1
    ? (newest - oldest) / (gallery.length - 1)
    : 0

  const byHour: Record<number, number> = {}
  for (const img of gallery) {
    const h = new Date(img.created * 1000).getHours()
    byHour[h] = (byHour[h] || 0) + 1
  }
  const peakHour = Object.entries(byHour).sort((a, b) => b[1] - a[1])[0]

  return { last24h, last7d, total: gallery.length, avgGap, peakHour: peakHour ? Number(peakHour[0]) : null }
}

export function ImageGalleryInsightsCard({ gallery, styles }: ImageGalleryInsightsCardProps) {
  const stats = computeStats(gallery)

  if (!stats || gallery.length === 0) return null

  return (
    <Card data-testid="image-gallery-insights">
      <CardHeader>
        <CardTitle className="text-base">Gallery Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total</div>
            <div className="text-lg font-semibold">{stats.total}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Last 24h</div>
            <div className="text-lg font-semibold">{stats.last24h}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Last 7d</div>
            <div className="text-lg font-semibold">{stats.last7d}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Avg Gap</div>
            <div className="text-lg font-semibold">
              {stats.avgGap < 60 ? '<1m' :
               stats.avgGap < 3600 ? `${Math.floor(stats.avgGap / 60)}m` :
               `${Math.floor(stats.avgGap / 3600)}h`}
            </div>
          </div>
        </div>
        {styles.length > 0 && (
          <div className="text-[11px] text-muted-foreground">
            {styles.length} style{styles.length !== 1 ? 's' : ''} available
            {stats.peakHour != null && (
              <span> · peak hour {stats.peakHour}:00</span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
