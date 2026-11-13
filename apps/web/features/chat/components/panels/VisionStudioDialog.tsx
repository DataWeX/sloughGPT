'use client'

import { memo } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@sloughgpt/strui'
import { Tabs } from '@sloughgpt/strui'
import { useVisionStudio } from './useVisionStudio'
import { AnalyzeTab } from './AnalyzeTab'
import { TrainTab } from './TrainTab'
import { GenerateTab } from './GenerateTab'
import { HistoryTab } from './HistoryTab'

interface VisionStudioDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string | null
  onGeneratedImage: (dataUrl: string, prompt: string) => void
  onSendText: (text: string) => void
  initialCaps?: {
    images_learned?: number
    trained?: boolean
    status?: string
    vocab_size?: number
    mean_accuracy?: number
  }
}

export const VisionStudioDialog = memo(function VisionStudioDialog({
  open, onOpenChange, sessionId, onGeneratedImage, onSendText, initialCaps,
}: VisionStudioDialogProps) {
  const vs = useVisionStudio(sessionId, onGeneratedImage, onSendText, initialCaps)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90dvh] overflow-hidden flex flex-col p-0 gap-0">
        <DialogHeader className="px-5 pt-4 pb-2 shrink-0">
          <DialogTitle>Vision Studio</DialogTitle>
        </DialogHeader>

        <Tabs
          value={vs.tab}
          onChange={vs.setTab}
          tabs={[
            { value: 'analyze', label: 'Analyze' },
            { value: 'train', label: 'Supervised Train' },
            { value: 'generate', label: 'Generate' },
            { value: 'history', label: 'History' },
          ]}
          className="mx-4"
        />

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {vs.tab === 'analyze' && <AnalyzeTab vs={vs} onSendText={onSendText} />}
          {vs.tab === 'train' && <TrainTab vs={vs} />}
          {vs.tab === 'generate' && <GenerateTab vs={vs} />}
          {vs.tab === 'history' && <HistoryTab vs={vs} />}
        </div>
      </DialogContent>
    </Dialog>
  )
})
