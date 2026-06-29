'use client'

import React, { useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ProgressBar } from '@/components/ui'
import { IconUpload } from '@/components/ui'
import type { TrainingStatus } from '@/lib/multimodal-controller'

interface BatchTrainingCardProps {
  batchUploading: boolean
  trainStatus: TrainingStatus | null
  onFileUpload: (files: File[]) => void
  onDirUpload: (dirPath: string) => void
}

export default function BatchTrainingCard({ batchUploading, trainStatus, onFileUpload, onDirUpload }: BatchTrainingCardProps) {
  const batchFileInputRef = useRef<HTMLInputElement>(null)
  const [batchDirPath, setBatchDirPath] = React.useState('')

  const handleBatchDir = () => {
    if (batchDirPath.trim()) onDirUpload(batchDirPath.trim())
  }

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Train with multiple images</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">Train on multiple images at once. Upload files or specify a server directory path.</p>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="h-8 text-xs" onClick={() => batchFileInputRef.current?.click()} disabled={batchUploading || trainStatus?.running}>
            <IconUpload className="h-3.5 w-3.5 mr-1" />
            {batchUploading ? 'Starting…' : 'Upload images'}
          </Button>
          <input ref={batchFileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={e => onFileUpload(Array.from(e.target.files || []))} />
        </div>
        <div className="flex items-center gap-2">
          <Input value={batchDirPath} onChange={e => setBatchDirPath(e.target.value)} placeholder="/path/to/images on server" className="h-8 text-xs flex-1" aria-label="Server directory path for batch training" />
          <Button size="sm" className="h-8 text-xs shrink-0" onClick={handleBatchDir} disabled={!batchDirPath.trim() || trainStatus?.running}>
            {trainStatus?.running ? 'Training…' : 'Train from directory'}
          </Button>
        </div>
        {trainStatus && trainStatus.running && (
          <div className="space-y-1 pt-2">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{trainStatus.completed}/{trainStatus.total} images</span>
              <span>{trainStatus.progress_pct}%</span>
            </div>
            <ProgressBar value={trainStatus.progress_pct} max={100} variant="default" />
            {trainStatus.current_image && (
              <p className="text-[10px] text-muted-foreground/60 truncate">Processing: {trainStatus.current_image}</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
