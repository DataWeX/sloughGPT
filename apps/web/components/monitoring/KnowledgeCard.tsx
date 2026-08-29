'use client'

import { memo } from 'react'
import { cn, Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'

interface KnowledgeCardProps {
  knowledgeStats: { total_items: number; topic_count: number; avg_importance: number; searchable: boolean } | null
  adapterStatus: { adapter_exists: boolean; fact_count: number; total_facts_available: number } | null
  loaded: boolean
}

export const KnowledgeCard = memo(function KnowledgeCard({ knowledgeStats, adapterStatus, loaded }: KnowledgeCardProps) {
  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Knowledge</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard label="Items" value={knowledgeStats ? knowledgeStats.total_items.toString() : (loaded ? '0' : <Skeleton className="h-5 w-10" />)} numeric />
          <StatCard label="Topics" value={knowledgeStats ? knowledgeStats.topic_count.toString() : (loaded ? '0' : <Skeleton className="h-5 w-10" />)} numeric />
          <StatCard label="Importance" value={knowledgeStats ? knowledgeStats.avg_importance.toFixed(2) : <Skeleton className="h-5 w-12" />} numeric />
          <StatCard
            label="AI training"
            value={!adapterStatus ? <Skeleton className="h-5 w-14" /> : <span className="font-mono">{adapterStatus.adapter_exists ? 'Trained' : 'Not'}</span>}
            icon={<span className={cn('inline-block w-2 h-2 rounded-full', !adapterStatus ? 'bg-warning' : adapterStatus.adapter_exists ? 'bg-success' : 'bg-muted-foreground/50')} />}
          />
        </KpiGrid>
        {adapterStatus && adapterStatus.adapter_exists && (
          <p className="text-[11px] text-muted-foreground mt-1.5 font-mono">
            {adapterStatus.fact_count} facts ({adapterStatus.total_facts_available} avail)
          </p>
        )}
      </CardContent>
    </Card>
  )
})
