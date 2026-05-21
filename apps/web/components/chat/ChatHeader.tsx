'use client'

import { Button } from '@/components/ui/button'
import { SearchInput } from '@/components/ui/input'
import { ModelStatusBar } from '@/components/InferenceStatusBar'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { getAgentName, AGENTS } from '@/lib/agents'
import type { AgentId } from '@/lib/agents'
import { IconMenu, IconChevronDown, IconCheck, IconPlus, IconSettings, IconMore, IconEye, IconDownload, IconTrash } from '@/components/ui'

interface ChatHeaderProps {
  health: ApiHealthSnapshot
  showSettings?: boolean
  showSidebar?: boolean
  onToggleSettings?: () => void
  onToggleSidebar?: () => void
  onNewChat?: () => void
  onViewConversation?: () => void
  onExportMd?: () => void
  onExportJson?: () => void
  onDeleteConversation?: () => void
  sessionCount?: number
  model?: string
  onModelChange?: (model: string) => void
  models?: string[]
  agent?: string
  onAgentChange?: (agent: string) => void
  temperature?: number
  maxTokens?: number
  onTemperatureChange?: (t: number) => void
  onMaxTokensChange?: (t: number) => void
  onSearch?: (query: string) => void
  searchQuery?: string
}

export function ChatHeader({
  health,
  showSidebar,
  onToggleSidebar,
  onNewChat,
  onViewConversation,
  onExportMd,
  onExportJson,
  onDeleteConversation,
  sessionCount = 0,
  model,
  onModelChange,
  models = [],
  agent = 'general',
  onAgentChange,
  onToggleSettings,
  onSearch,
  searchQuery = '',
}: ChatHeaderProps) {
  return (
    <AppRouteHeader
      sticky
      className="py-3 !border-transparent bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
      left={
        <AppRouteHeaderLead
          title={
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={onToggleSidebar}
                className="p-1.5"
                title="Toggle sidebar"
                aria-label="Toggle sidebar"
                aria-expanded={showSidebar}
              >
                <IconMenu className="h-4 w-4" />
              </Button>
              <h1 className="text-lg font-semibold text-foreground">Chat</h1>
              {sessionCount > 0 && (
                <span className="font-mono text-xs text-muted-foreground">{sessionCount} conversations</span>
              )}
            </div>
          }
          subtitle={
            <div className="hidden sm:block">
              <ModelStatusBar health={health} />
            </div>
          }
        />
      }
      right={
        <div className="flex items-center gap-1.5">
          {onSearch && (
            <div className="relative hidden sm:block">
              <SearchInput
                value={searchQuery}
                onChange={onSearch}
                placeholder="Search..."
                className="w-32 lg:w-40"
              />
            </div>
          )}
          {agent && onAgentChange && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="select" size="sm" className="text-xs max-w-[80px] sm:max-w-[100px]">
                  <span className="truncate">{getAgentName(agent)}</span>
                  <IconChevronDown className="h-3 w-3 opacity-60 shrink-0" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[140px]">
                {(Object.keys(AGENTS) as AgentId[]).map((id) => (
                  <DropdownMenuItem
                    key={id}
                    onSelect={() => onAgentChange(id)}
                    className="justify-between"
                  >
                    <span className="truncate">{getAgentName(id)}</span>
                    {id === agent && <IconCheck className="h-3.5 w-3.5 text-primary shrink-0" />}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          {model && onModelChange && models.length > 0 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="select" size="sm" className="text-xs max-w-[70px] sm:max-w-[90px] font-mono">
                  <span className="truncate">{model}</span>
                  <IconChevronDown className="h-3 w-3 opacity-60 shrink-0" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[140px]">
                {models.map((m) => (
                  <DropdownMenuItem
                    key={m}
                    onSelect={() => onModelChange(m)}
                    className="justify-between font-mono"
                  >
                    <span className="truncate">{m}</span>
                    {m === model && <IconCheck className="h-3.5 w-3.5 text-primary shrink-0" />}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={onNewChat}
            className="p-1.5"
            title="New chat (Ctrl+N)"
            aria-label="Start new chat"
          >
            <IconPlus className="h-4 w-4" />
          </Button>
          {onToggleSettings && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onToggleSettings}
              className="p-1.5"
              title="Settings"
              aria-label="Open settings"
            >
              <IconSettings className="h-4 w-4" />
            </Button>
          )}
          {onViewConversation && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="p-1.5"
                  title="Conversation menu"
                  aria-label="Open conversation menu"
                >
                  <IconMore className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[160px]">
                <DropdownMenuItem onSelect={onViewConversation}>
                  <IconEye className="mr-2 h-4 w-4" />
                  View conversation
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={onExportMd}>
                  <IconDownload className="mr-2 h-4 w-4" />
                  Export as Markdown
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={onExportJson}>
                  <IconDownload className="mr-2 h-4 w-4" />
                  Export as JSON
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={onToggleSettings}>
                  <IconSettings className="mr-2 h-4 w-4" />
                  Settings
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={onDeleteConversation} className="text-destructive focus:text-destructive">
                  <IconTrash className="mr-2 h-4 w-4" />
                  Delete conversation
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      }
    />
  )
}