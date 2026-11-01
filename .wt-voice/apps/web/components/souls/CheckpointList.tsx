'use client'

import Link from 'next/link'
import { cn, Card, CardHeader, CardTitle, CardContent, Button, SearchInput, Spinner } from '@sloughgpt/strui'
import { IconRefresh, IconDownload, IconTrash } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'
import { verdictBadge } from './soul-helpers'
import { formatShortDate } from '@/lib/time-format'

interface CheckpointListProps {
  checkpoints: Checkpoint[]
  searchQuery: string
  onSearchChange: (query: string) => void
  loadingCheckpoint: string | null
  onLoad: (name: string) => void
  onDownload: (name: string) => void
  onDelete: (name: string) => void
  onInfo: (name: string) => void
  onRefresh: () => void
}

export function CheckpointList({
  checkpoints,
  searchQuery,
  onSearchChange,
  loadingCheckpoint,
  onLoad,
  onDownload,
  onDelete,
  onInfo,
  onRefresh,
}: CheckpointListProps) {
  const filtered = checkpoints.filter(cp =>
    cp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    cp.soul?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Checkpoints ({checkpoints.length})</CardTitle>
        <div className="flex gap-2">
          <SearchInput
            value={searchQuery}
            onChange={onSearchChange}
            placeholder="Search checkpoints..."
            className="max-w-xs"
          />
          <Button size="sm" variant="ghost" onClick={onRefresh} aria-label="Refresh">
            <IconRefresh className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {filtered.length === 0 ? (
          <div className="text-center py-8 space-y-2">
            <p className="text-sm text-muted-foreground">No checkpoints found.</p>
            <div className="text-xs text-muted-foreground">
              <Link href="/training" prefetch={false} className="text-primary hover:underline">Train a model</Link>
              {' '}to create checkpoints.
            </div>
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {filtered.map(cp => (
              <div key={cp.name} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2.5 text-sm group hover:bg-muted/50 transition-colors">
                <div className="flex-1 min-w-0 cursor-pointer" onClick={() => onInfo(cp.name)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onInfo(cp.name); } }} role="button" tabIndex={0}>
                  <div className="flex items-center gap-2">
                    <span className="font-medium truncate">{cp.name}</span>
                    {cp.verdict && (
                      <span className={cn('text-xs px-1.5 py-0.5 rounded font-medium', verdictBadge(cp.verdict).className)}>
                        {verdictBadge(cp.verdict).label}
                      </span>
                    )}
                    {cp.is_loaded && (
                      <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">loaded</span>
                    )}
                    {cp.model_type && (
                      <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{cp.model_type}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                    {cp.soul && <span>{cp.soul}</span>}
                    {cp.loss != null && <span>loss {cp.loss.toFixed(3)}</span>}
                    {cp.size_mb != null && <span>{cp.size_mb.toFixed(1)} MB</span>}
                    {cp.training_dataset && <span>· {cp.training_dataset.split('/').pop()}</span>}
                    {cp.training_duration_s != null && cp.training_duration_s > 0 && <span>· {cp.training_duration_s.toFixed(0)}s</span>}
                    {cp.born_at && <span>· {formatShortDate(cp.born_at)}</span>}
                  </div>
                  {cp.perplexity_delta != null && cp.perplexity_delta !== 0 && (
                    <div className="flex items-center gap-3 text-xs mt-0.5">
                      <span className={cp.perplexity_delta < 0 ? 'text-success' : 'text-destructive'}>
                        PPL {cp.perplexity_delta > 0 ? '+' : ''}{cp.perplexity_delta.toFixed(3)}
                      </span>
                      {cp.bleu_delta != null && cp.bleu_delta !== 0 && (
                        <span className={cp.bleu_delta > 0 ? 'text-success' : 'text-destructive'}>
                          BLEU {cp.bleu_delta > 0 ? '+' : ''}{cp.bleu_delta.toFixed(3)}
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                  <Button size="sm" variant="ghost" onClick={() => onLoad(cp.name)} disabled={loadingCheckpoint === cp.name}>
                    {loadingCheckpoint === cp.name ? (
                      <Spinner size="sm" />
                    ) : 'Load'}
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => onDownload(cp.name)} aria-label="Download checkpoint">
                    <IconDownload className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-destructive"
                    onClick={() => onDelete(cp.name)}
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
  )
}
