'use client'

import { cn } from '@/lib/cn'

interface Agent {
  id: string
  name: string
  description: string
  instructions: string
}

interface AgentsTabProps {
  agents: Agent[]
  currentAgent: Agent | null
  onSelectAgent: (agent: Agent | null) => void
}

export function AgentsTab({ agents, currentAgent, onSelectAgent }: AgentsTabProps) {
  return (
    <div className="space-y-1">
      {agents.length === 0 ? (
        <p className="text-xs text-muted-foreground">No agents configured. Create one in the Agents page.</p>
      ) : (
        <>
          <button
            onClick={() => onSelectAgent(null)}
            className={cn(
              'w-full text-left px-2 py-1.5 rounded text-xs transition-colors',
              currentAgent === null ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted',
            )}
          >
            <span className="text-[10px] text-muted-foreground block">Default (no agent)</span>
            <span>Direct chat</span>
          </button>
          {agents.map(agent => (
            <button
              key={agent.id}
              onClick={() => onSelectAgent(agent)}
              className={cn(
                'w-full text-left px-2 py-1.5 rounded text-xs transition-colors',
                currentAgent?.id === agent.id ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted',
              )}
            >
              <span className="font-medium">{agent.name}</span>
              {agent.description && (
                <span className="text-[10px] text-muted-foreground block truncate">{agent.description}</span>
              )}
            </button>
          ))}
        </>
      )}
    </div>
  )
}
