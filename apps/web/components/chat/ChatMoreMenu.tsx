'use client'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { IconMore, IconSettings } from '@/components/ui'
import { cn } from '@/lib/cn'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

const STATUS_COLORS: Record<string, string> = {
  ok: 'bg-success',
  degraded: 'bg-warning',
  offline: 'bg-destructive',
  loading: 'bg-muted-foreground/30 animate-pulse',
}
const STATUS_LABELS: Record<string, string> = {
  ok: 'Model loaded',
  degraded: 'Server degraded',
  offline: 'Server offline',
  loading: 'Connecting...',
}

export function ChatMoreMenu() {
  const ctx = useChatToolbarContext()
  const { onVoiceMode, onToggleTools, onExportMarkdown, onCopyMarkdown, onSaveAsDataset, onSystemPrompt, onSearchConversations, hasMessages, messageCount } = ctx.actions
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-2.5 text-xs gap-1.5 rounded-lg" aria-label="More options">
          <IconMore className="h-3.5 w-3.5" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[180px]">
        <div className="px-3 py-2 space-y-1">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${STATUS_COLORS[ctx.health.status]}`} />
            <span>{ctx.health.summary || STATUS_LABELS[ctx.health.status]}</span>
          </div>
          {messageCount > 0 && (
            <div className="text-[10px] text-muted-foreground">{messageCount} messages</div>
          )}
          {ctx.knowledge.count > 0 && (
            <div className="text-[10px] text-muted-foreground">{ctx.knowledge.count} knowledge facts</div>
          )}
          {ctx.actions.bookmarkCount > 0 && (
            <div className="text-[10px] text-muted-foreground">{ctx.actions.bookmarkCount} bookmarks</div>
          )}
        </div>

        <DropdownMenuSeparator />

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
          <span className="relative mr-2 h-4 w-4 inline-flex items-center justify-center">
            <span className={cn(
              'w-2.5 h-2.5 rounded-sm transition-colors border',
              ctx.localEngine.useLocal ? 'bg-primary border-primary' : 'bg-transparent border-muted-foreground/40'
            )}>
              {ctx.localEngine.useLocal && (
                <svg className="w-2.5 h-2.5 text-primary-foreground" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
              )}
            </span>
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
          <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
          Search Conversations
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onSystemPrompt}>
          <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>
          System Prompt
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={onExportMarkdown} disabled={!hasMessages}>
          <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 11l5 5 5-5M12 4v12" /></svg>
          Export Markdown
        </DropdownMenuItem>
        {onCopyMarkdown && (
          <DropdownMenuItem onSelect={onCopyMarkdown} disabled={!hasMessages}>
            <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" /></svg>
            Copy to clipboard
          </DropdownMenuItem>
        )}
        {onSaveAsDataset && (
          <DropdownMenuItem onSelect={onSaveAsDataset} disabled={!hasMessages}>
            <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" /></svg>
            Save as dataset
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
