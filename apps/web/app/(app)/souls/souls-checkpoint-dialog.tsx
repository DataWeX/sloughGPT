'use client'

/** Checkpoint detail dialog: verdict, loss/delta stats, load and download. */
import {
  Button,
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  IconDownload,
  cn,
} from '@sloughgpt/strui'
import { formatDateTimeFull } from '@/lib/time-format'
import { verdictBadge } from './souls-format'
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function CheckpointDetailDialog({ wb }: { wb: SoulsWorkbench }) {
  const {
    checkpointDetail,
    currentSoul,
    handleDownloadCheckpoint,
    handleLoadCheckpoint,
    setCheckpointDetail,
  } = wb

  return (
    <Dialog
      open={checkpointDetail !== null}
      onOpenChange={(open) => {
        if (!open) setCheckpointDetail(null)
      }}
    >
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {checkpointDetail?.name}
            {checkpointDetail?.verdict && (
              <span
                className={cn(
                  'text-xs px-1.5 py-0.5 rounded font-medium',
                  verdictBadge(checkpointDetail.verdict).className,
                )}
              >
                {verdictBadge(checkpointDetail.verdict).label}
              </span>
            )}
          </DialogTitle>
        </DialogHeader>
        {checkpointDetail && (
          <div className="space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-2">
              {checkpointDetail.soul && (
                <div>
                  <span className="text-muted-foreground">Soul:</span>{' '}
                  <span className="font-medium">{checkpointDetail.soul}</span>
                </div>
              )}
              {checkpointDetail.model_type && (
                <div>
                  <span className="text-muted-foreground">Type:</span>{' '}
                  <span className="font-medium">{checkpointDetail.model_type}</span>
                </div>
              )}
              {checkpointDetail.loss != null && (
                <div>
                  <span className="text-muted-foreground">Loss:</span>{' '}
                  <span className="font-mono">{checkpointDetail.loss.toFixed(4)}</span>
                </div>
              )}
              {checkpointDetail.size_mb != null && (
                <div>
                  <span className="text-muted-foreground">Size:</span>{' '}
                  <span className="font-mono">{checkpointDetail.size_mb.toFixed(1)} MB</span>
                </div>
              )}
              {checkpointDetail.epochs != null && (
                <div>
                  <span className="text-muted-foreground">Epochs:</span>{' '}
                  <span className="font-mono">{checkpointDetail.epochs}</span>
                </div>
              )}
              {checkpointDetail.steps != null && (
                <div>
                  <span className="text-muted-foreground">Steps:</span>{' '}
                  <span className="font-mono">{checkpointDetail.steps}</span>
                </div>
              )}
              {checkpointDetail.training_dataset && (
                <div className="col-span-2">
                  <span className="text-muted-foreground">Dataset:</span>{' '}
                  <span className="font-mono">{checkpointDetail.training_dataset}</span>
                </div>
              )}
              {checkpointDetail.born_at && (
                <div className="col-span-2">
                  <span className="text-muted-foreground">Created:</span>{' '}
                  <span className="font-medium">
                    {formatDateTimeFull(checkpointDetail.born_at)}
                  </span>
                </div>
              )}
            </div>
            {checkpointDetail.perplexity_delta != null &&
              checkpointDetail.perplexity_delta !== 0 && (
                <div className="flex gap-4 pt-1 border-t border-border/30">
                  <span
                    className={
                      checkpointDetail.perplexity_delta < 0 ? 'text-success' : 'text-destructive'
                    }
                  >
                    Perplexity: {checkpointDetail.perplexity_delta > 0 ? '+' : ''}
                    {checkpointDetail.perplexity_delta.toFixed(3)}
                  </span>
                  {checkpointDetail.bleu_delta != null && checkpointDetail.bleu_delta !== 0 && (
                    <span
                      className={
                        checkpointDetail.bleu_delta > 0 ? 'text-success' : 'text-destructive'
                      }
                    >
                      BLEU: {checkpointDetail.bleu_delta > 0 ? '+' : ''}
                      {checkpointDetail.bleu_delta.toFixed(3)}
                    </span>
                  )}
                </div>
              )}
            <div className="flex justify-end gap-2 pt-2">
              <Button size="sm" variant="outline" onClick={() => setCheckpointDetail(null)}>
                Close
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  if (checkpointDetail) handleDownloadCheckpoint(checkpointDetail.name)
                }}
              >
                <IconDownload className="h-3.5 w-3.5 mr-1" /> Download
              </Button>
              {currentSoul !== checkpointDetail.soul && (
                <Button
                  size="sm"
                  onClick={() => {
                    if (checkpointDetail) {
                      handleLoadCheckpoint(checkpointDetail.name)
                      setCheckpointDetail(null)
                    }
                  }}
                >
                  Load
                </Button>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
