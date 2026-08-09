'use client'

import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogFooter, DialogTitle, DialogDescription } from '@sloughgpt/strui'
import { Checkbox } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { sessionStore } from '@/lib/session-store'

interface DownloadDialogProps {
  open: boolean
  pendingDownload: string | null
  modelInfoMap: Record<string, { size_gb?: number }>
  onConfirm: (modelId: string) => void
  onCancel: () => void
}

export function DownloadDialog({
  open,
  pendingDownload,
  modelInfoMap,
  onConfirm,
  onCancel,
}: DownloadDialogProps) {
  const [skipConfirm, setSkipConfirm] = useState(false)

  if (!pendingDownload) return null

  const info = modelInfoMap[pendingDownload]
  const sizeText = info?.size_gb ? `${info.size_gb.toFixed(1)} GB` : '? GB'
  const modelName = pendingDownload.includes('/') ? pendingDownload.split('/').pop() : pendingDownload

  return (
    <Dialog open={open} onOpenChange={(open) => { if (!open) onCancel() }}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Download model?</DialogTitle>
          <DialogDescription>
            Download {modelName} ({sizeText}) from HuggingFace? This will use your internet connection.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="flex-col sm:flex-row gap-3 sm:items-center">
          <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer select-none">
            <Checkbox
              checked={skipConfirm}
              onCheckedChange={(c) => setSkipConfirm(c === true)}
            />
            <span>Don&apos;t ask again this session</span>
          </label>
          <div className="flex gap-2 ml-auto">
            <Button variant="outline" size="sm" onClick={onCancel}>
              Cancel
            </Button>
            <Button size="sm" onClick={() => {
              if (skipConfirm) sessionStore.setApproved(pendingDownload)
              onCancel()
              onConfirm(pendingDownload)
            }}>
              Download
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
