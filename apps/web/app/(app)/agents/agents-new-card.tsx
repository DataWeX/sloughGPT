'use client'

/**
 * Creator card: the new-agent form with template shortcuts and validation.
 */
import type { AgentsWorkbench } from './useAgentsWorkbench'
import { AGENT_TEMPLATES, AVAILABLE_TOOLS } from './agents-constants'
import { Button, Card, CardContent, CardHeader, CardTitle, Input, cn } from '@sloughgpt/strui'
import { IconPlus } from '@/components/icons/NavIcons'
import { SectionErrorBoundary } from '@/components/SectionErrorBoundary'

export function NewAgentCard({
  newName,
  setNewName,
  newDesc,
  setNewDesc,
  newInstructions,
  setNewInstructions,
  newTools,
  setNewTools,
  creating,
  createErrors,
  toggleTool,
  handleCreate,
  setCreateErrors,
}: {
  newName: AgentsWorkbench['newName']
  setNewName: AgentsWorkbench['setNewName']
  newDesc: AgentsWorkbench['newDesc']
  setNewDesc: AgentsWorkbench['setNewDesc']
  newInstructions: AgentsWorkbench['newInstructions']
  setNewInstructions: AgentsWorkbench['setNewInstructions']
  newTools: AgentsWorkbench['newTools']
  setNewTools: AgentsWorkbench['setNewTools']
  creating: AgentsWorkbench['creating']
  createErrors: AgentsWorkbench['createErrors']
  toggleTool: AgentsWorkbench['toggleTool']
  handleCreate: AgentsWorkbench['handleCreate']
  setCreateErrors: AgentsWorkbench['setCreateErrors']
}) {
  return (
    <SectionErrorBoundary sectionName="Agent creator">
      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">New Agent</CardTitle>
        </CardHeader>
        <CardContent className="px-2.5 pb-2.5 space-y-3">
          <div>
            <p className="text-[10px] font-medium text-muted-foreground mb-2">
              Start from a template
            </p>
            <div className="flex flex-wrap gap-1.5">
              {AGENT_TEMPLATES.map((t) => (
                <button
                  key={t.name}
                  type="button"
                  onClick={() => {
                    setNewName(t.name)
                    setNewDesc(t.desc)
                    setNewInstructions(t.instructions)
                    setNewTools([...t.tools])
                  }}
                  className="rounded-full px-3 py-1 text-[10px] font-medium border border-border/40 bg-muted/20 text-muted-foreground hover:bg-primary/10 hover:text-primary hover:border-primary/30 focus:ring-2 focus:ring-primary/30 transition-colors"
                >
                  {t.name}
                </button>
              ))}
            </div>
          </div>
          <div>
            <Input
              id="create-agent-name"
              placeholder="Name"
              aria-label="Agent name"
              value={newName}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                setNewName(e.target.value)
                if (createErrors.name) setCreateErrors((prev) => ({ ...prev, name: undefined }))
              }}
              className={createErrors.name ? 'border-destructive ring-destructive/20' : ''}
              aria-invalid={!!createErrors.name}
              aria-describedby={createErrors.name ? 'create-name-error' : undefined}
            />
            {createErrors.name && (
              <p id="create-name-error" className="text-[10px] text-destructive mt-1" role="alert">
                {createErrors.name}
              </p>
            )}
          </div>
          <Input
            placeholder="Description (optional)"
            value={newDesc}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewDesc(e.target.value)}
          />
          <textarea
            className="w-full rounded-lg border border-input bg-background px-3 py-2 text-[11px] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[80px]"
            placeholder="Instructions — how the agent should behave"
            value={newInstructions}
            onChange={(e) => setNewInstructions(e.target.value)}
            aria-label="Agent instructions"
          />
          <div className="flex flex-wrap gap-1">
            {AVAILABLE_TOOLS.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => toggleTool(t, newTools, setNewTools)}
                aria-pressed={newTools.includes(t)}
                className={cn(
                  'rounded-full px-3 py-1 text-[10px] font-medium border transition-colors',
                  newTools.includes(t)
                    ? 'bg-primary text-primary-foreground border-primary'
                    : 'bg-background text-muted-foreground border-border hover:border-primary/50',
                )}
              >
                {t.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
          <Button onClick={handleCreate} disabled={creating || !newName.trim()}>
            <IconPlus className="h-4 w-4 mr-1" />
            {creating ? 'Creating...' : 'Create Agent'}
          </Button>
        </CardContent>
      </Card>
    </SectionErrorBoundary>
  )
}
