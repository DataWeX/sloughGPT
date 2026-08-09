'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import type { KnowledgeItem, KnowledgeStats } from '@/lib/knowledge-controller'

interface KnowledgeCategoryChartProps {
  items: KnowledgeItem[]
  stats: KnowledgeStats | null
}

function topicColor(index: number): string {
  const colors = [
    'bg-primary', 'bg-accent', 'bg-success', 'bg-warning',
    'bg-primary/60', 'bg-accent/60', 'bg-success/60', 'bg-warning/60',
  ]
  return colors[index % colors.length]
}

function computeTopicDistribution(items: KnowledgeItem[]): Array<{ topic: string; count: number; pct: number }> {
  if (items.length === 0) return []
  const counts: Record<string, number> = {}
  for (const item of items) {
    const t = item.topic || 'uncategorized'
    counts[t] = (counts[t] || 0) + 1
  }
  return Object.entries(counts)
    .map(([topic, count]) => ({ topic, count, pct: count / items.length }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8)
}

function computeImportanceDistribution(items: KnowledgeItem[]): Array<{ label: string; count: number; pct: number }> {
  if (items.length === 0) return []
  const buckets = [
    { label: 'Critical', min: 0.8, max: 1.01 },
    { label: 'High', min: 0.6, max: 0.8 },
    { label: 'Medium', min: 0.4, max: 0.6 },
    { label: 'Low', min: 0, max: 0.4 },
  ]
  return buckets.map(({ label, min, max }) => {
    const count = items.filter(i => i.importance >= min && i.importance < max).length
    return { label, count, pct: count / items.length }
  }).filter(b => b.count > 0)
}

export function KnowledgeCategoryChart({ items, stats }: KnowledgeCategoryChartProps) {
  const topics = computeTopicDistribution(items)
  const importance = computeImportanceDistribution(items)

  if (items.length === 0) return null

  const avgImportance = stats?.avg_importance ?? (
    items.reduce((s, i) => s + i.importance, 0) / items.length
  )

  return (
    <Card data-testid="knowledge-category-chart">
      <CardHeader>
        <CardTitle className="text-base">Knowledge Breakdown</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-2">
              By Topic ({topics.length})
            </div>
            <div className="space-y-1.5">
              {topics.map((t, idx) => (
                <div key={t.topic} className="flex items-center gap-2">
                  <span className="text-[11px] text-muted-foreground w-24 truncate">{t.topic}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted/50 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${topicColor(idx)}`}
                      style={{ width: `${t.pct * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{t.count}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-2">
              By Importance
            </div>
            <div className="space-y-1.5">
              {importance.map(i => (
                <div key={i.label} className="flex items-center gap-2">
                  <span className="text-[11px] text-muted-foreground w-24">{i.label}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted/50 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        i.label === 'Critical' ? 'bg-success' :
                        i.label === 'High' ? 'bg-primary' :
                        i.label === 'Medium' ? 'bg-warning' :
                        'bg-muted-foreground/40'
                      }`}
                      style={{ width: `${i.pct * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{i.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-4 pt-3 border-t border-border/30 flex items-center gap-4 text-[10px] text-muted-foreground">
          <span>{items.length} items</span>
          <span>{topics.length} topics</span>
          <span>avg importance {(avgImportance * 100).toFixed(0)}%</span>
          {stats?.searchable && <span className="text-success">searchable</span>}
        </div>
      </CardContent>
    </Card>
  )
}
