'use client'

/** KPI strip: remember/topics/care/searchable, skeleton while loading. */
import { Card, CardContent } from '@sloughgpt/strui'
import { KnowledgeStatsSkeleton } from '@/components/ui/PageSkeletons'
import type { KnowledgeWorkbench } from './useKnowledgeWorkbench'

export function KnowledgeStatsStrip({ wb }: { wb: KnowledgeWorkbench }) {
  const { loading, stats } = wb

  return (
    <>
      {loading && !stats ? (
        <KnowledgeStatsSkeleton />
      ) : stats ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Card>
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                Things I remember
              </p>
              <p className="text-xl font-semibold mt-1">{stats.total_items}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                Topics
              </p>
              <p className="text-xl font-semibold mt-1">{stats.topic_count}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                How much I care
              </p>
              <p className="text-xl font-semibold mt-1">
                {stats.avg_importance >= 0.7
                  ? 'A lot'
                  : stats.avg_importance >= 0.4
                    ? 'Somewhat'
                    : 'A little'}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                Searchable
              </p>
              <p className="text-xl font-semibold mt-1">{stats.searchable ? 'Yes' : 'No'}</p>
            </CardContent>
          </Card>
        </div>
      ) : null}
    </>
  )
}
