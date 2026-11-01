'use client'

import { cn, Button } from '@sloughgpt/strui'
import { IconUpload, IconX } from '@sloughgpt/strui'
import type { UseVisionStudioReturn } from './useVisionStudio'

interface TrainTabProps {
  vs: UseVisionStudioReturn
}

export function TrainTab({ vs }: TrainTabProps) {
  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        Train the vision model with supervised labels for better accuracy.
        Provide a ground truth caption to improve the model&apos;s understanding.
      </p>

      <div
        onClick={() => vs.fileInputRef.current?.click()}
        className={cn(
          'border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
          vs.previewUrl ? 'border-primary/30 bg-primary/3' : 'border-border/50 hover:border-primary/40',
        )}
      >
        {vs.previewUrl ? (
          <div className="relative inline-block">
            <img src={vs.previewUrl} alt="Training image" className="max-h-40 rounded object-contain" />
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); vs.clearPreview() }}
              className="absolute -top-2 -right-2 h-5 w-5 flex items-center justify-center rounded-full bg-background border border-border/50 shadow-sm"
              aria-label="Remove"
            >
              <IconX className="h-3 w-3" />
            </button>
          </div>
        ) : (
          <>
            <IconUpload className="h-6 w-6 mx-auto mb-1 text-muted-foreground" />
            <p className="text-xs text-muted-foreground">Select an image for training</p>
          </>
        )}
        <input ref={vs.fileInputRef} type="file" accept="image/*" className="hidden" onChange={vs.handleFileSelect} aria-label="Upload image for training" />
      </div>

      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground">Ground truth label (caption)</label>
        <input
          className="w-full px-3 py-2 text-sm border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
          placeholder="e.g., 'a red car on a sunny road'"
          value={vs.trainLabel}
          onChange={(e) => vs.setTrainLabel(e.target.value)}
          aria-label="Ground truth label for training image"
          disabled={!vs.previewUrl || vs.trainLoading}
        />
      </div>

      <Button
        className="w-full"
        disabled={!vs.previewUrl || vs.trainLoading}
        onClick={vs.handleTrainWithLabel}
      >
        {vs.trainLoading ? 'Training...' : 'Train with label'}
      </Button>

      {vs.trainResult && (
        <div className="p-3 rounded-lg bg-muted/30 border border-border/40 space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">BLEU accuracy:</span>
            <span className={cn('font-medium', vs.trainResult.accuracy >= 80 ? 'text-success' : vs.trainResult.accuracy >= 50 ? 'text-warning' : 'text-muted-foreground')}>
              {vs.trainResult.accuracy.toFixed(1)}%
            </span>
          </div>
          <div className="text-xs text-muted-foreground">
            Caption: <span className="text-foreground">{vs.trainResult.caption}</span>
          </div>
        </div>
      )}
    </div>
  )
}
