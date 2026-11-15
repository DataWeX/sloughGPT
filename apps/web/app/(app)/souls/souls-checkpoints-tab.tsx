'use client'

/** Checkpoints tab: search, verdict badges, load/download/delete actions. */
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  SearchInput,
  Spinner,
  TabsContent,
  IconDownload,
  IconRefresh,
  IconTrash,
  cn,
} from '@sloughgpt/strui'
import Link from '@/vite/next-compat/link'
import { formatShortDate } from '@/lib/time-format'
import { verdictBadge } from './souls-format'
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function CheckpointsTab({ wb }: { wb: SoulsWorkbench }) {
  const {
    checkpoints,
    filteredCheckpoints,
    handleCheckpointInfo,
    handleDownloadCheckpoint,
    handleLoadCheckpoint,
    handleRefresh,
    loadingCheckpoint,
    searchQuery,
    setDeleteTarget,
    setDeleteType,
    setSearchQuery,
  } = wb

  return (
    <TabsContent value="checkpoints" className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Checkpoints ({checkpoints.length})</CardTitle>
          <div className="flex gap-2">
            <SearchInput
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search checkpoints..."
              className="max-w-xs"
            />
            <Button size="sm" variant="ghost" onClick={handleRefresh} aria-label="Refresh">
              <IconRefresh className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {filteredCheckpoints.length === 0 ? (
            <div className="text-center py-8 space-y-2">
              <p className="text-sm text-muted-foreground">No checkpoints found.</p>
              <div className="text-xs text-muted-foreground">
                <Link href="/training" prefetch={false} className="text-primary hover:underline">
                  Train a model
                </Link>{' '}
                to create checkpoints.
              </div>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredCheckpoints.map((cp) => (
                <div
                  key={cp.name}
                  className="flex items-center justify-between rounded-md border border-border/40 px-3 py-2.5 text-sm group hover:bg-muted/50 transition-colors"
                >
                  <div
                    className="flex-1 min-w-0 cursor-pointer"
                    onClick={() => handleCheckpointInfo(cp.name)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        handleCheckpointInfo(cp.name)
                      }
                    }}
                    role="button"
                    tabIndex={0}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate">{cp.name}</span>
                      {cp.verdict && (
                        <span
                          className={cn(
                            'text-xs px-1.5 py-0.5 rounded font-medium',
                            verdictBadge(cp.verdict).className,
                          )}
                        >
                          {verdictBadge(cp.verdict).label}
                        </span>
                      )}
                      {cp.is_loaded && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                          loaded
                        </span>
                      )}
                      {cp.model_type && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                          {cp.model_type}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                      {cp.soul && <span>{cp.soul}</span>}
                      {cp.loss != null && <span>loss {cp.loss.toFixed(3)}</span>}
                      {cp.size_mb != null && <span>{cp.size_mb.toFixed(1)} MB</span>}
                      {cp.training_dataset && <span>· {cp.training_dataset.split('/').pop()}</span>}
                      {cp.training_duration_s != null && cp.training_duration_s > 0 && (
                        <span>· {cp.training_duration_s.toFixed(0)}s</span>
                      )}
                      {cp.born_at && <span>· {formatShortDate(cp.born_at)}</span>}
                    </div>
                    {cp.perplexity_delta != null && cp.perplexity_delta !== 0 && (
                      <div className="flex items-center gap-3 text-xs mt-0.5">
                        <span
                          className={cp.perplexity_delta < 0 ? 'text-success' : 'text-destructive'}
                        >
                          PPL {cp.perplexity_delta > 0 ? '+' : ''}
                          {cp.perplexity_delta.toFixed(3)}
                        </span>
                        {cp.bleu_delta != null && cp.bleu_delta !== 0 && (
                          <span className={cp.bleu_delta > 0 ? 'text-success' : 'text-destructive'}>
                            BLEU {cp.bleu_delta > 0 ? '+' : ''}
                            {cp.bleu_delta.toFixed(3)}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleLoadCheckpoint(cp.name)}
                      disabled={loadingCheckpoint === cp.name}
                    >
                      {loadingCheckpoint === cp.name ? <Spinner size="sm" /> : 'Load'}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDownloadCheckpoint(cp.name)}
                      aria-label="Download checkpoint"
                    >
                      <IconDownload className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive"
                      onClick={() => {
                        setDeleteTarget(cp.name)
                        setDeleteType('checkpoint')
                      }}
                      aria-label="Delete checkpoint"
                    >
                      <IconTrash className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </TabsContent>
  )
}
