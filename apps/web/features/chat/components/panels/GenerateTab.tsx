'use client'

import { Button } from '@sloughgpt/strui'
import { IconSend } from '@sloughgpt/strui'
import type { UseVisionStudioReturn } from './useVisionStudio'

interface GenerateTabProps {
  vs: UseVisionStudioReturn
}

export function GenerateTab({ vs }: GenerateTabProps) {
  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input
          className="flex-1 px-3 py-2 text-sm border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
          placeholder="Describe an image to generate..."
          value={vs.genPrompt}
          onChange={(e) => vs.setGenPrompt(e.target.value)}
          aria-label="Image generation prompt"
          onKeyDown={(e) => { if (e.key === 'Enter') vs.handleGenerateImage() }}
          disabled={vs.genLoading}
        />
        <Button onClick={vs.handleGenerateImage} disabled={vs.genLoading || !vs.genPrompt.trim()}>
          {vs.genLoading ? 'Generating...' : 'Generate'}
        </Button>
      </div>

      {vs.genError && (
        <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">{vs.genError}</div>
      )}

      {vs.genResult && (
        <div className="space-y-2">
          <div className="relative rounded-lg overflow-hidden border border-border/50 bg-muted/5">
            <img src={vs.genResult} alt="Generated" className="w-full max-h-72 object-contain" />
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={vs.handleSendGeneratedImage}>
              <IconSend className="h-3.5 w-3.5 mr-1" />
              Send to chat
            </Button>
            <Button size="sm" variant="ghost" onClick={() => vs.setGenResult?.(null)}>
              Dismiss
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
