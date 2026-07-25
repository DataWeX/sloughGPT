'use client'

import { cn, DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconMore, IconSettings, IconSearch, IconCopy, IconExport, IconDocument } from '@sloughgpt/strui'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

export function ChatMoreMenu() {
  const ctx = useChatToolbarContext()
  const { onVoiceMode, onToggleTools, onExportMarkdown, onCopyMarkdown, onSaveAsDataset, onSystemPrompt, onSearchConversations, hasMessages } = ctx.actions
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-2.5 text-xs gap-1.5 rounded-lg" aria-label="More options">
          <IconMore className="h-3.5 w-3.5" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[180px]">
        {ctx.agent.agents.length > 0 && (
          <>
            <div className="px-3 py-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Agent {ctx.agent.current && `— ${ctx.agent.current.name}`}
            </div>
            {ctx.agent.agents.map(a => (
              <DropdownMenuItem key={a.id || a.name} onSelect={() => ctx.agent.onSelect(a)}>
                <span className={cn('mr-2 h-4 w-4 inline-flex items-center justify-center',
                  ctx.agent.current?.id === a.id || ctx.agent.current?.name === a.name
                    ? 'text-primary' : 'text-muted-foreground/40'
                )}>
                  {ctx.agent.current?.id === a.id || ctx.agent.current?.name === a.name ? (
                    <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                  ) : (
                    <div className="w-3.5 h-3.5 rounded-full border border-current" />
                  )}
                </span>
                {a.name}
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
          </>
        )}

        <DropdownMenuItem onSelect={ctx.localEngine.onToggle}>
          <span className={cn(
            'mr-2 h-4 w-4 inline-flex items-center justify-center rounded-sm transition-colors border',
            ctx.localEngine.useLocal ? 'bg-primary border-primary' : 'border-muted-foreground/40'
          )}>
            {ctx.localEngine.useLocal && (
              <svg className="w-2.5 h-2.5 text-primary-foreground" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
            )}
          </span>
          <span className="flex-1">Local Engine</span>
          {ctx.localEngine.loading && <span className="w-3 h-3 rounded-full bg-warning animate-pulse ml-auto" />}
        </DropdownMenuItem>

        <DropdownMenuSeparator />

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
        <DropdownMenuItem onSelect={onSearchConversations}>
          <IconSearch className="mr-2 h-4 w-4" />
          Search
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onSystemPrompt}>
          <IconDocument className="mr-2 h-4 w-4" />
          System Prompt
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        <DropdownMenuItem onSelect={onExportMarkdown} disabled={!hasMessages}>
          <IconExport className="mr-2 h-4 w-4" />
          Export Markdown
        </DropdownMenuItem>
        {onCopyMarkdown && (
          <DropdownMenuItem onSelect={onCopyMarkdown} disabled={!hasMessages}>
            <IconCopy className="mr-2 h-4 w-4" />
            Copy
          </DropdownMenuItem>
        )}
        {onSaveAsDataset && (
          <DropdownMenuItem onSelect={onSaveAsDataset} disabled={!hasMessages}>
            <IconDocument className="mr-2 h-4 w-4" />
            Save as Dataset
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
