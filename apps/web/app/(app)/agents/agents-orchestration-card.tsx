'use client'

/**
 * Multi-agent orchestration: goal, agent picker, phase/task progress.
 */
import type { AgentsWorkbench } from './useAgentsWorkbench'
import { Button, Card, CardContent, CardHeader, CardTitle, Input, cn } from '@sloughgpt/strui'
import { SectionErrorBoundary } from '@/components/SectionErrorBoundary'
import { StatusBanner } from '@/components/composed/StatusBanner'

export function OrchestrationCard({
  agents,
  orchGoal,
  setOrchGoal,
  orchContext,
  setOrchContext,
  orchRunning,
  orchPhase,
  orchTasks,
  orchTaskStatuses,
  orchLevel,
  orchTotalLevels,
  orchResponse,
  orchError,
  orchAgentIds,
  setOrchAgentIds,
  orchErrors,
  handleOrchestrate,
  setOrchErrors,
  setOrchResponse,
  setOrchError,
  setOrchPhase,
  setOrchTasks,
  setOrchTaskStatuses,
}: {
  agents: AgentsWorkbench['agents']
  orchGoal: AgentsWorkbench['orchGoal']
  setOrchGoal: AgentsWorkbench['setOrchGoal']
  orchContext: AgentsWorkbench['orchContext']
  setOrchContext: AgentsWorkbench['setOrchContext']
  orchRunning: AgentsWorkbench['orchRunning']
  orchPhase: AgentsWorkbench['orchPhase']
  orchTasks: AgentsWorkbench['orchTasks']
  orchTaskStatuses: AgentsWorkbench['orchTaskStatuses']
  orchLevel: AgentsWorkbench['orchLevel']
  orchTotalLevels: AgentsWorkbench['orchTotalLevels']
  orchResponse: AgentsWorkbench['orchResponse']
  orchError: AgentsWorkbench['orchError']
  orchAgentIds: AgentsWorkbench['orchAgentIds']
  setOrchAgentIds: AgentsWorkbench['setOrchAgentIds']
  orchErrors: AgentsWorkbench['orchErrors']
  handleOrchestrate: AgentsWorkbench['handleOrchestrate']
  setOrchErrors: AgentsWorkbench['setOrchErrors']
  setOrchResponse: AgentsWorkbench['setOrchResponse']
  setOrchError: AgentsWorkbench['setOrchError']
  setOrchPhase: AgentsWorkbench['setOrchPhase']
  setOrchTasks: AgentsWorkbench['setOrchTasks']
  setOrchTaskStatuses: AgentsWorkbench['setOrchTaskStatuses']
}) {
  return (
    <SectionErrorBoundary sectionName="Orchestrator panel">
      <Card>
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
              value={orchGoal}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                setOrchGoal(e.target.value)
                if (orchErrors.goal) setOrchErrors((prev) => ({ ...prev, goal: undefined }))
              }}
              onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                if (e.key === 'Enter' && !e.shiftKey && orchGoal.trim() && !orchRunning) {
                  e.preventDefault()
                  handleOrchestrate()
                }
              }}
              className={orchErrors.goal ? 'border-destructive ring-destructive/20' : ''}
              aria-invalid={!!orchErrors.goal}
              aria-describedby={orchErrors.goal ? 'orch-goal-error' : undefined}
            />
            {orchErrors.goal && (
              <p id="orch-goal-error" className="text-[10px] text-destructive mt-1" role="alert">
                {orchErrors.goal}
              </p>
            )}
          </div>
          <textarea
            className="w-full rounded-lg border border-input bg-background px-3 py-2 text-[11px] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[60px]"
            placeholder="Additional context (optional)"
            value={orchContext}
            onChange={(e) => setOrchContext(e.target.value)}
            aria-label="Orchestration context"
          />
          {agents.length > 0 && (
            <div>
              <p className="text-[10px] font-medium text-muted-foreground mb-1.5">
                Agents (optional — leave empty for all)
              </p>
              <div className="flex flex-wrap gap-1.5">
                {agents.map((a) => (
                  <button
                    key={a.id}
                    type="button"
                    onClick={() => {
                      setOrchAgentIds((prev) =>
                        prev.includes(a.id) ? prev.filter((id) => id !== a.id) : [...prev, a.id],
                      )
                    }}
                    className={cn(
                      'rounded-full px-2.5 py-1 text-[10px] font-medium border transition-colors',
                      orchAgentIds.includes(a.id)
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
            <Button
              onClick={handleOrchestrate}
              disabled={orchRunning || !orchGoal.trim()}
              className="h-7 text-[11px]"
            >
              {orchRunning ? 'Orchestrating...' : 'Orchestrate'}
            </Button>
            {(orchResponse || orchError) && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setOrchResponse(null)
                  setOrchError(null)
                  setOrchPhase('')
                  setOrchTasks([])
                  setOrchTaskStatuses({})
                }}
                className="h-6 text-[10px]"
              >
                Clear
              </Button>
            )}
          </div>

          {/* Phase indicator */}
          {orchPhase && (
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
              <span
                className={cn(
                  'inline-block h-2 w-2 rounded-full',
                  orchPhase === 'COMPLETE'
                    ? 'bg-success'
                    : orchPhase === 'ERROR'
                      ? 'bg-destructive'
                      : 'bg-warning animate-pulse',
                )}
              />
              {orchPhase === 'PLAN' && 'Planning subtasks...'}
              {orchPhase === 'EXECUTE' && `Executing level ${orchLevel}/${orchTotalLevels}...`}
              {orchPhase === 'COMPOSE' && 'Composing final response...'}
              {orchPhase === 'COMPLETE' && 'Complete'}
              {orchPhase === 'ERROR' && 'Failed'}
            </div>
          )}

          {/* Task status */}
          {orchTasks.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-[10px] font-medium text-muted-foreground">
                Tasks ({orchTasks.length})
              </p>
              {orchTasks.map((task) => {
                const status = orchTaskStatuses[task.id] || 'pending'
                return (
                  <div
                    key={task.id}
                    className="flex items-center gap-2 rounded-lg border border-border/40 px-2.5 py-2 text-[11px]"
                  >
                    <span
                      className={cn(
                        'inline-block h-2 w-2 shrink-0 rounded-full',
                        status === 'completed'
                          ? 'bg-success'
                          : status === 'live'
                            ? 'bg-warning animate-pulse'
                            : status === 'failed'
                              ? 'bg-destructive'
                              : 'bg-muted-foreground/30',
                      )}
                    />
                    <span className="font-medium text-[10px] min-w-[64px] text-muted-foreground">
                      {task.agent}
                    </span>
                    <span className="flex-1 truncate">{task.description}</span>
                    {task.depends_on && task.depends_on.length > 0 && (
                      <span className="text-[10px] text-muted-foreground/50">
                        after: {task.depends_on.join(', ')}
                      </span>
                    )}
                  </div>
                )
              })}
            </div>
          )}

          {/* Final response */}
          {orchResponse && (
            <div className="rounded-lg border border-border/40 bg-muted/30 p-3">
              <p className="text-[10px] font-medium text-muted-foreground mb-1">Result</p>
              <p className="text-[11px] whitespace-pre-wrap">{orchResponse}</p>
            </div>
          )}

          {/* Error */}
          {orchError && <StatusBanner variant="error" message={orchError} dismissible={false} />}
        </CardContent>
      </Card>
    </SectionErrorBoundary>
  )
}
