'use client'

import { cn, Button } from '@sloughgpt/strui'
import { IconUpload, IconX, IconRefresh, IconSend, IconDownload, IconAlert } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import type { UseVisionStudioReturn } from './useVisionStudio'

interface AnalyzeTabProps {
  vs: UseVisionStudioReturn
  onSendText: (text: string) => void
}

export function AnalyzeTab({ vs, onSendText }: AnalyzeTabProps) {
  return (
    <div className="space-y-4">
      <div
        ref={vs.dropRef}
        onDragOver={(e) => { e.preventDefault(); vs.setDragOver(true) }}
        onDragLeave={() => vs.setDragOver(false)}
        onDrop={vs.handleDrop}
        onClick={() => vs.fileInputRef.current?.click()}
        className={cn(
          'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
          vs.dragOver ? 'border-primary bg-primary/5' : 'border-border/50 hover:border-primary/40',
        )}
      >
        <IconUpload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm font-medium mb-1">
          {vs.previewUrl ? 'Change image' : 'Drop an image here or browse'}
        </p>
        <p className="text-xs text-muted-foreground">Supports JPG, PNG, WebP</p>
        <input
          ref={vs.fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={vs.handleFileSelect}
          aria-label="Upload image for analysis"
        />
      </div>

      {vs.previewUrl && (
        <div className="relative rounded-lg overflow-hidden border border-border/50">
          <img src={vs.previewUrl} alt="Preview" className="w-full max-h-64 object-contain bg-muted/5" />
          <button
            type="button"
            onClick={vs.clearPreview}
            className="absolute top-2 right-2 h-7 w-7 flex items-center justify-center rounded-full bg-background/80 hover:bg-background border border-border/50 shadow-sm"
            aria-label="Clear preview"
          >
            <IconX className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {vs.analyzeLoading && (
        <div className="space-y-2">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      )}

      {vs.analyzeError && (
        <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
          {vs.analyzeError}
        </div>
      )}

      {vs.analyzeResult && (
        <div className="space-y-3">
          <div className="p-3 rounded-lg bg-muted/30 border border-border/40">
            <div className="text-[10px] text-muted-foreground font-medium mb-1 uppercase tracking-wider">Caption</div>
            <div className="text-sm leading-relaxed">{vs.analyzeResult.caption}</div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
              <div className="text-sm font-semibold">{vs.analyzeResult.confidence.toFixed(2)}</div>
              <div className="text-[10px] text-muted-foreground">Confidence</div>
            </div>
            <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
              <div className="text-sm font-semibold">{vs.analyzeResult.images_learned}</div>
              <div className="text-[10px] text-muted-foreground">Images learned</div>
            </div>
            <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
              <div className={cn(
                'text-sm font-semibold',
                vs.analyzeResult.mean_accuracy >= 80 ? 'text-success' : vs.analyzeResult.mean_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground',
              )}>
                {vs.analyzeResult.mean_accuracy.toFixed(1)}%
              </div>
              <div className="text-[10px] text-muted-foreground">Mean accuracy</div>
            </div>
          </div>

          {vs.analyzeResult.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {vs.analyzeResult.tags.map((tag) => (
                <span key={tag} className="px-1.5 py-0.5 rounded-full bg-muted/50 text-[10px] border border-border/30">{tag}</span>
              ))}
            </div>
          )}

          {(vs.analyzeResult.confidence < 0.3 || vs.analyzeResult.caption === '[caption failed]') && (
            <div className="p-3 rounded-lg bg-warning/10 border border-warning/20 text-sm space-y-2">
              <div className="flex items-center gap-2 text-warning font-medium">
                <IconAlert className="h-4 w-4 shrink-0" />
                Low confidence — consider retrying
              </div>
              <p className="text-xs text-muted-foreground">
                The model isn&apos;t confident about this result. Try again — the model may produce a better result on a second pass.
              </p>
              <Button size="sm" variant="outline" onClick={vs.retryAnalyze} disabled={vs.retryLoading} className="h-7 text-xs">
                <IconRefresh className="h-3 w-3 mr-1" />
                {vs.retryLoading ? 'Retrying…' : 'Retry analysis'}
              </Button>
            </div>
          )}
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => onSendText(vs.analyzeResult!.caption)}
              disabled={!vs.analyzeResult.caption || vs.analyzeResult.caption === '[caption failed]'}
            >
              <IconSend className="h-3.5 w-3.5 mr-1" />
              Send caption to chat
            </Button>
            {vs.analyzeResult.caption && vs.analyzeResult.caption !== '[caption failed]' && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  navigator.clipboard.writeText(vs.analyzeResult!.caption)
                }}
              >
                <IconDownload className="h-3.5 w-3.5 mr-1" />
                Copy caption
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
