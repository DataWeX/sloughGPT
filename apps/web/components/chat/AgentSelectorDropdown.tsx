'use client'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { IconChevronDown, IconCheck } from '@/components/ui'
import type { AgentDef } from '@/lib/agents'

const AGENT_ICONS: Record<string, string> = {
  coder: '💻',
  writer: '✍️',
  researcher: '🔬',
  analyst: '📊',
}

function getAgentIcon(agent: AgentDef): string {
  return agent.icon || AGENT_ICONS[agent.id] || '💬'
}

interface AgentSelectorDropdownProps {
  agents: AgentDef[]
  currentAgent: AgentDef | null
  onSelect: (agent: AgentDef) => void
}

export function AgentSelectorDropdown({ agents, currentAgent, onSelect }: AgentSelectorDropdownProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-1.5 sm:px-2.5 text-xs gap-1 sm:gap-1.5 rounded-lg border border-border/40 hover:border-border/60" title={currentAgent?.description}>
          <span className="text-sm shrink-0">{currentAgent ? getAgentIcon(currentAgent) : '💬'}</span>
          <span className="hidden sm:inline truncate max-w-[64px]">{currentAgent?.name || 'Role'}</span>
          <IconChevronDown className="hidden sm:block h-2.5 w-2.5 opacity-40 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[180px]">
        {agents.map((a) => (
          <DropdownMenuItem
            key={a.id}
            onSelect={() => onSelect(a)}
            className="justify-between text-xs"
          >
            <div className="flex items-center gap-2">
              <span className="text-sm">{getAgentIcon(a)}</span>
              <div>
                <div>{a.name}</div>
                <div className="text-[10px] text-muted-foreground">{a.description}</div>
              </div>
            </div>
            {currentAgent?.id === a.id && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
