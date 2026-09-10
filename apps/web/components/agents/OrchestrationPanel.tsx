'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatusBanner, cn } from '@sloughgpt/strui'

interface Agent {
  id: string
  name: string
}

interface OrchestrateTask {
  id: string
  agent: string
  description: string
  depends_on?: string[]
}

interface OrchestrationPanelProps {
  agents: Agent[]
  goal: string
  context: string
  running: boolean
  phase: string
  tasks: OrchestrateTask[]
  taskStatuses: Record<string, string>
  level: number
  totalLevels: number
  response: string | null
  error: string | null
  selectedAgentIds: string[]
  errors: { goal?: string; context?: string }
  onGoalChange: (value: string) => void
  onContextChange: (value: string) => void
  onAgentToggle: (agentId: string) => void
  onOrchestrate: () => void
  onClear: () => void
}

export function OrchestrationPanel({
  agents,
  goal,
  context,
  running,
  phase,
  tasks,
  taskStatuses,
  level,
  totalLevels,
  response,
  error,
  selectedAgentIds,
  errors,
  onGoalChange,
  onContextChange,
  onAgentToggle,
  onOrchestrate,
  onClear,
}: OrchestrationPanelProps) {
  return (
    <Card data-testid="orchestration-panel">
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Multi-Agent Orchestration</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5 space-y-3">
        <p className="text-[10px] text-muted-foreground">
          Decompose a goal into subtasks and execute them across multiple agents in parallel.
        </p>
        <div>
          <Input
            id="orch-goal"
            placeholder="Goal — e.g. research transformers and write a summary"
            value={goal}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => onGoalChange(e.target.value)}
            onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
              if (e.key === 'Enter' && !e.shiftKey && goal.trim() && !running) {
                e.preventDefault()
                onOrchestrate()
              }
            }}
            className={errors.goal ? 'border-destructive ring-destructive/20' : ''}
            aria-invalid={!!errors.goal}
            aria-describedby={errors.goal ? 'orch-goal-error' : undefined}
          />
          {errors.goal && (
            <p id="orch-goal-error" className="text-[10px] text-destructive mt-1" role="alert">
              {errors.goal}
            </p>
          )}
        </div>
        <textarea
          className="w-full rounded-lg border border-input bg-background px-3 py-2 text-[11px] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[60px]"
          placeholder="Additional context (optional)"
          value={context}
          onChange={e => onContextChange(e.target.value)}
          aria-label="Orchestration context"
        />
        {agents.length > 0 && (
          <div>
            <p className="text-[10px] font-medium text-muted-foreground mb-1.5">Agents (optional — leave empty for all)</p>
            <div className="flex flex-wrap gap-1.5" data-testid="orch-agent-list">
              {agents.map(a => (
                <button
                  key={a.id}
                  type="button"
                  onClick={() => onAgentToggle(a.id)}
                  className={cn(
                    'rounded-full px-2.5 py-1 text-[10px] font-medium border transition-colors',
                    selectedAgentIds.includes(a.id)
                      ? 'bg-primary text-primary-foreground border-primary'
                      : 'bg-background text-muted-foreground border-border hover:border-primary/50',
                  )}
                >
                  {a.name}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="flex gap-2">
          <Button onClick={onOrchestrate} disabled={running || !goal.trim()} className="h-7 text-[11px]">
            {running ? 'Orchestrating...' : 'Orchestrate'}
          </Button>
          {(response || error) && (
            <Button size="sm" variant="ghost" onClick={onClear} className="h-6 text-[10px]">
              Clear
            </Button>
          )}
        </div>

        {phase && (
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground" data-testid="phase-indicator">
            <span className={cn(
              'inline-block h-2 w-2 rounded-full',
              phase === 'COMPLETE' ? 'bg-success' : phase === 'ERROR' ? 'bg-destructive' : 'bg-warning animate-pulse',
            )} />
            {phase === 'PLAN' && 'Planning subtasks...'}
            {phase === 'EXECUTE' && `Executing level ${level}/${totalLevels}...`}
            {phase === 'COMPOSE' && 'Composing final response...'}
            {phase === 'COMPLETE' && 'Complete'}
            {phase === 'ERROR' && 'Failed'}
          </div>
        )}

        {tasks.length > 0 && (
          <div className="space-y-1.5" data-testid="orch-tasks">
            <p className="text-[10px] font-medium text-muted-foreground">Tasks ({tasks.length})</p>
            {tasks.map(task => {
              const status = taskStatuses[task.id] || 'pending'
              return (
                <div key={task.id} className="flex items-center gap-2 rounded-lg border border-border/40 px-2.5 py-2 text-[11px]">
                  <span className={cn(
                    'inline-block h-2 w-2 shrink-0 rounded-full',
                    status === 'completed' ? 'bg-success' : status === 'in_progress' ? 'bg-warning animate-pulse' : status === 'failed' ? 'bg-destructive' : 'bg-muted-foreground/30',
                  )} />
                  <span className="font-medium text-[10px] min-w-[64px] text-muted-foreground">{task.agent}</span>
                  <span className="flex-1 truncate">{task.description}</span>
                  {task.depends_on && task.depends_on.length > 0 && (
                    <span className="text-[10px] text-muted-foreground/50">after: {task.depends_on.join(', ')}</span>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {response && (
          <div className="rounded-lg border border-border/40 bg-muted/30 p-3" data-testid="orch-response">
            <p className="text-[10px] font-medium text-muted-foreground mb-1">Result</p>
            <p className="text-[11px] whitespace-pre-wrap">{response}</p>
          </div>
        )}

        {error && (
          <StatusBanner variant="error" message={error} dismissible={false} />
        )}
      </CardContent>
    </Card>
  )
}
