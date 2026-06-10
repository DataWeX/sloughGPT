'use client'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { IconMore, IconSettings } from '@/components/ui'
import type { ChatMessage } from './ChatMessages'

interface ChatMoreMenuProps {
  onVoiceMode: () => void
  onToggleTools: () => void
  onExportMarkdown: () => void
  hasMessages: boolean
}

export function ChatMoreMenu({ onVoiceMode, onToggleTools, onExportMarkdown, hasMessages }: ChatMoreMenuProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-2.5 text-xs gap-1.5 rounded-lg" aria-label="More options">
          <IconMore className="h-3.5 w-3.5" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[160px]">
        <DropdownMenuItem onSelect={onVoiceMode}>
          <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
          </svg>
          Voice Mode
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onToggleTools}>
          <IconSettings className="mr-2 h-4 w-4" />
          Tools Panel
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onExportMarkdown} disabled={!hasMessages}>
          <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 11l5 5 5-5M12 4v12" /></svg>
          Export Markdown
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
