'use client'

import { cn, DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconMore, IconSettings, IconSearch, IconCopy, IconExport, IconDocument, IconPlus, IconCheck, IconBrain, IconMicFilled } from '@sloughgpt/strui'
import { useChatToolbarContext } from '@/features/chat/contexts/ChatToolbarContext'

function ShortcutHint({ keys }: { keys: string }) {
  return <span className="ml-auto text-[9px] text-muted-foreground/50 font-mono">{keys}</span>
}

export function ChatMoreMenu() {
  const ctx = useChatToolbarContext()
  const { onVoiceMode, onToggleTools, onExportMarkdown, onCopyMarkdown, onSaveAsDataset, onSystemPrompt, onSearchConversations, hasMessages } = ctx.actions
  const onNewChat = ctx.conversations.onNewChat
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-2.5 text-xs gap-1.5 rounded-lg" aria-label="More options">
          <IconMore className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[200px]">
        <DropdownMenuItem onSelect={onNewChat}>
          <IconPlus className="mr-2 h-4 w-4" />
          <span className="flex-1">New Conversation</span>
          <ShortcutHint keys="Ctrl+N" />
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        {ctx.agent.agents.length > 0 && (
          <>
            <div className="px-3 py-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Agent {ctx.agent.current && `— ${ctx.agent.current.name}`}
            </div>
            {ctx.agent.agents.map(a => {
              const isActive = ctx.agent.current?.id === a.id || ctx.agent.current?.name === a.name
              return (
                <DropdownMenuItem key={a.id || a.name} onSelect={() => ctx.agent.onSelect(a)} className={`flex items-start gap-2 px-3 py-2 ${isActive ? 'bg-primary/8' : ''}`}>
                  <span className={`mt-0.5 h-4 w-4 shrink-0 inline-flex items-center justify-center ${isActive ? 'text-primary' : 'text-muted-foreground/40'}`}>
                    {isActive ? (
                      <IconCheck className="h-3.5 w-3.5" />
                    ) : (
                      <div className="w-3.5 h-3.5 rounded-full border border-current" />
                    )}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="font-medium truncate">{a.name}</div>
                    {a.description && (
                      <div className="text-[10px] text-muted-foreground truncate mt-0.5">{a.description}</div>
                    )}
                    {a.capabilities && a.capabilities.length > 0 && (
                      <span className="text-[9px] text-muted-foreground mt-1 inline-block">{a.capabilities.length} capabilities</span>
                    )}
                  </div>
                </DropdownMenuItem>
              )
            })}
            <DropdownMenuSeparator />
          </>
        )}

        <DropdownMenuItem onSelect={ctx.localEngine.onToggle}>
          <span className={cn(
            'mr-2 h-4 w-4 inline-flex items-center justify-center rounded-sm transition-colors border',
            ctx.localEngine.useLocal ? 'bg-primary border-primary' : 'border-muted-foreground/40'
          )}>
            {ctx.localEngine.useLocal && (
              <IconCheck className="w-2.5 h-2.5 text-primary-foreground" />
            )}
          </span>
          <span className="flex-1">Local Engine</span>
          {ctx.localEngine.loading && <span className="w-3 h-3 rounded-full bg-warning animate-pulse ml-auto" />}
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        <DropdownMenuItem onSelect={onVoiceMode}>
          <IconMicFilled className="mr-2 h-4 w-4" />
          Voice Mode
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onToggleTools}>
          <IconSettings className="mr-2 h-4 w-4" />
          <span className="flex-1">Tools Panel</span>
          <ShortcutHint keys="Ctrl+." />
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onSearchConversations}>
          <IconSearch className="mr-2 h-4 w-4" />
          <span className="flex-1">Search</span>
          <ShortcutHint keys="Ctrl+F" />
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onSystemPrompt}>
          <IconDocument className="mr-2 h-4 w-4" />
          System Prompt
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        <DropdownMenuItem onSelect={onExportMarkdown} disabled={!hasMessages}>
          <IconExport className="mr-2 h-4 w-4" />
          <span className="flex-1">Export Markdown</span>
          <ShortcutHint keys="Ctrl+Shift+E" />
        </DropdownMenuItem>
        {onCopyMarkdown && (
          <DropdownMenuItem onSelect={onCopyMarkdown} disabled={!hasMessages}>
            <IconCopy className="mr-2 h-4 w-4" />
            <span className="flex-1">Copy</span>
            <ShortcutHint keys="Ctrl+Shift+C" />
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
