'use client'

/** Advanced fold: category chart, spaced review, memory training, adapter and RAG. */
import { Button, Card, CardContent, Chip, FoldSection, Spinner, cn } from '@sloughgpt/strui'
import {
  KnowledgeAdapterSkeleton,
  KnowledgeRAGSkeleton,
  KnowledgeTopicsSkeleton,
} from '@/components/ui/PageSkeletons'
import { KnowledgeCategoryChart } from '@/components/knowledge/KnowledgeCategoryChart'
import { SpacedReviewCard } from '@/components/knowledge/SpacedReviewCard'
import { MS_PER_SECOND } from '@/lib/format-bytes'
import type { KnowledgeWorkbench } from './useKnowledgeWorkbench'

export function KnowledgeAdvancedSection({ wb }: { wb: KnowledgeWorkbench }) {
  const {
    activeTopic,
    adapterStatus,
    adapterTraining,
    addToast,
    handleRAGClear,
    handleRAGLoadMoreDocs,
    handleRAGSync,
    handleTrainAdapter,
    items,
    loading,
    ragClearing,
    ragDocs,
    ragDocsTotal,
    ragLoadingMore,
    ragStats,
    ragSyncing,
    setActiveTopic,
    setShowRagDocs,
    showRagDocs,
    stats,
    topics,
  } = wb

  return (
    <FoldSection heading="Advanced — memory & topics">
      {items.length > 0 && <KnowledgeCategoryChart items={items} stats={stats} />}
      <SpacedReviewCard addToast={addToast} />
      {loading && !adapterStatus ? (
        <KnowledgeAdapterSkeleton />
      ) : adapterStatus ? (
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                  Memory Training
                </p>
                {adapterStatus.adapter_exists ? (
                  <span className="text-xs px-1.5 py-0.5 rounded-full bg-success/15 text-success font-medium">
                    Ready
                  </span>
                ) : (
                  <span className="text-xs px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
                    Not trained
                  </span>
                )}
              </div>
              <Button
                size="sm"
                variant={adapterStatus.adapter_exists ? 'outline' : 'default'}
                className="h-7 text-xs px-2.5"
                onClick={handleTrainAdapter}
                disabled={adapterTraining || items.length === 0}
              >
                {adapterTraining ? (
                  <span className="flex items-center gap-1">
                    <Spinner size="xs" />
                    Learning...
                  </span>
                ) : adapterStatus.adapter_exists ? (
                  'Retrain'
                ) : (
                  'Train memory'
                )}
              </Button>
            </div>
            <div className="flex gap-3 mt-2 text-xs text-muted-foreground">
              <span>{adapterStatus.fact_count} things learned</span>
              {adapterStatus.trained_at && (
                <span>
                  Last trained{' '}
                  {new Date(adapterStatus.trained_at * MS_PER_SECOND).toLocaleDateString()}
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      ) : null}
      {loading && !ragStats ? (
        <KnowledgeRAGSkeleton />
      ) : ragStats ? (
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                  Deep Memory
                </p>
                <span className="text-xs px-1.5 py-0.5 rounded-full bg-success/15 text-success font-medium">
                  {ragStats.total_chunks} pieces
                </span>
              </div>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs px-2.5"
                  onClick={handleRAGSync}
                  disabled={ragSyncing}
                >
                  {ragSyncing ? (
                    <span className="flex items-center gap-1">
                      <Spinner size="xs" />
                      Syncing...
                    </span>
                  ) : (
                    'Sync'
                  )}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs px-2.5"
                  onClick={() => setShowRagDocs(!showRagDocs)}
                >
                  {showRagDocs ? 'Hide' : 'Sources'}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs px-2.5 text-destructive hover:text-destructive"
                  onClick={handleRAGClear}
                  disabled={ragClearing}
                >
                  {ragClearing ? 'Clearing...' : 'Clear'}
                </Button>
              </div>
            </div>
            <div className="flex gap-3 mt-2 text-xs text-muted-foreground">
              <span>{ragStats.total_documents} sources</span>
              <span>{ragStats.total_chunks} pieces indexed</span>
            </div>
            {showRagDocs && ragDocs.length > 0 && (
              <>
                <div className="mt-3 space-y-1 max-h-48 overflow-y-auto">
                  {ragDocs.map((doc, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between text-xs py-1 border-b border-border/40 last:border-0"
                    >
                      <div className="flex-1 min-w-0">
                        <span className="text-foreground truncate block">
                          {(doc.metadata?.source as string) || 'unknown'}
                        </span>
                        <span className="text-muted-foreground">{doc.num_chunks} pieces</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                  <span>
                    Showing {ragDocs.length} of {ragDocsTotal}
                  </span>
                  {ragDocs.length < ragDocsTotal && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-xs"
                      onClick={handleRAGLoadMoreDocs}
                      disabled={ragLoadingMore}
                    >
                      {ragLoadingMore ? 'Loading…' : 'Load more'}
                    </Button>
                  )}
                </div>
              </>
            )}
            {showRagDocs && ragDocs.length === 0 && (
              <p className="mt-3 text-xs text-muted-foreground">No sources yet.</p>
            )}
          </CardContent>
        </Card>
      ) : null}
      {loading && topics.length === 0 ? (
        <KnowledgeTopicsSkeleton />
      ) : topics.length > 0 ? (
        <Card>
          <CardContent className="p-3">
            <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium mb-2">
              What we talk about
            </p>
            <div className="space-y-1.5">
              {topics.slice(0, 8).map((t) => {
                const pct = stats ? Math.round((t.count / stats.total_items) * 100) : 0
                return (
                  <div key={t.name} className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setActiveTopic(activeTopic === t.name ? null : t.name)}
                      className={cn(
                        'text-xs w-24 text-left truncate',
                        activeTopic === t.name
                          ? 'text-primary font-medium'
                          : 'text-muted-foreground hover:text-foreground',
                      )}
                    >
                      {t.name}
                    </button>
                    <div className="flex-1 h-3 bg-muted/50 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary/40 rounded-full"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground w-8 text-right">{t.count}</span>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      ) : null}
      {topics.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          <Chip
            label={`All (${items.length})`}
            onClick={() => setActiveTopic(null)}
            className={cn(
              'text-xs px-2 py-0.5 rounded-full cursor-pointer',
              activeTopic === null
                ? 'bg-primary/15 text-primary border-primary/30'
                : 'bg-muted text-muted-foreground border-border/40',
            )}
          />
          {topics.map((t) => (
            <Chip
              key={t.name}
              label={`${t.name} (${t.count})`}
              onClick={() => setActiveTopic(activeTopic === t.name ? null : t.name)}
              className={cn(
                'text-xs px-2 py-0.5 rounded-full cursor-pointer',
                activeTopic === t.name
                  ? 'bg-primary/15 text-primary border-primary/30'
                  : 'bg-muted text-muted-foreground border-border/40',
              )}
            />
          ))}
        </div>
      )}
    </FoldSection>
  )
}
