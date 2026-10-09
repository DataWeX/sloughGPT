'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input, Checkbox, cn } from '@sloughgpt/strui'
import { Plus } from 'lucide-react'

interface AgentCreateCardProps {
  name: string
  instructions: string
  tools: string[]
  availableTools: string[]
  loading: boolean
  onNameChange: (value: string) => void
  onInstructionsChange: (value: string) => void
  onToolsChange: (tools: string[]) => void
  onCreate: () => void
}

export function AgentCreateCard({
  name,
  instructions,
  tools,
  availableTools,
  loading,
  onNameChange,
  onInstructionsChange,
  onToolsChange,
  onCreate,
}: AgentCreateCardProps) {
  const toggleTool = (tool: string) => {
    onToolsChange(
      tools.includes(tool)
        ? tools.filter(t => t !== tool)
        : [...tools, tool]
    )
  }

  return (
    <Card data-testid="agent-create-card">
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Plus className="h-4 w-4" />
          Create Agent
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <Input
            value={name}
            onChange={e => onNameChange(e.target.value)}
            placeholder="Agent name"
            maxLength={100}
            data-testid="agent-name"
          />
          <textarea
            value={instructions}
            onChange={e => onInstructionsChange(e.target.value)}
            placeholder="Agent instructions..."
            rows={4}
            className="w-full rounded border border-border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            data-testid="agent-instructions"
          />
          {availableTools.length > 0 && (
            <div className="space-y-1.5">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Tools</div>
              <div className="flex flex-wrap gap-2">
                {availableTools.map(tool => (
                  <label key={tool} className="flex items-center gap-1.5 text-xs cursor-pointer">
                    <Checkbox
                      checked={tools.includes(tool)}
                      onCheckedChange={() => toggleTool(tool)}
                    />
                    {tool}
                  </label>
                ))}
              </div>
            </div>
          )}
          <Button
            onClick={onCreate}
            disabled={!name.trim() || loading}
            data-testid="agent-create-btn"
          >
            {loading ? 'Creating...' : 'Create'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
