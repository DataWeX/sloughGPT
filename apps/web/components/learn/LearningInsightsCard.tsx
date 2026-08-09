'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface KnowledgeFact {
  content: string
  topic: string
  source: string
  importance: number
}

interface LearningInsightsCardProps {
  facts: KnowledgeFact[]
}

export function LearningInsightsCard({ facts }: LearningInsightsCardProps) {
  if (facts.length === 0) return null

  const byTopic: Record<string, number> = {}
  const bySource: Record<string, number> = {}
  let totalImportance = 0
  let highImportance = 0
  let avgContentLen = 0

  for (const f of facts) {
    const topic = f.topic || 'untagged'
    const source = f.source || 'unknown'
    byTopic[topic] = (byTopic[topic] ?? 0) + 1
    bySource[source] = (bySource[source] ?? 0) + 1
    totalImportance += f.importance
    if (f.importance >= 0.8) highImportance++
    avgContentLen += f.content.length
  }

  avgContentLen = Math.round(avgContentLen / facts.length)
  const avgImportance = (totalImportance / facts.length).toFixed(2)

  const sortedTopics = Object.entries(byTopic).sort((a, b) => b[1] - a[1])
  const sortedSources = Object.entries(bySource).sort((a, b) => b[1] - a[1])

  return (
    <Card data-testid="learning-insights">
      <CardHeader>
        <CardTitle className="text-base">Learning Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-2 mb-3">
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Facts</div>
            <div className="text-sm font-mono font-medium">{facts.length}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Avg Importance</div>
            <div className="text-sm font-mono font-medium">{avgImportance}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">High Priority</div>
            <div className="text-sm font-mono font-medium text-success">{highImportance}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Avg Length</div>
            <div className="text-sm font-mono font-medium">{avgContentLen}ch</div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="text-[10px] text-muted-foreground mb-1">By Topic</div>
            <div className="space-y-1">
              {sortedTopics.slice(0, 5).map(([topic, count]) => (
                <div key={topic} className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground truncate">{topic}</span>
                  <span className="font-mono">{count}</span>
                </div>
              ))}
              {sortedTopics.length > 5 && (
                <div className="text-[9px] text-muted-foreground">+{sortedTopics.length - 5} more</div>
              )}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground mb-1">By Source</div>
            <div className="space-y-1">
              {sortedSources.slice(0, 5).map(([source, count]) => (
                <div key={source} className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground truncate">{source}</span>
                  <span className="font-mono">{count}</span>
                </div>
              ))}
              {sortedSources.length > 5 && (
                <div className="text-[9px] text-muted-foreground">+{sortedSources.length - 5} more</div>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
