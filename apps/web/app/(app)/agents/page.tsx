'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { EmptyCard, KpiGrid, StatCard, IconRefresh } from '@sloughgpt/strui'
import { IconPlus, IconTrash, IconClock } from '@/components/icons/NavIcons'
import { agentsController, type Agent, type OrchestrateTask, type AgentRun } from '@/lib/agents-controller'
import { useToastStore } from '@/lib/toast-store'

const AVAILABLE_TOOLS = ['web_search', 'code_execution', 'file_read', 'knowledge_retrieval', 'image_analysis', 'data_analysis']

export default function AgentsPage() {
  const addToast = useToastStore(s => s.addToast)
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)

  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newInstructions, setNewInstructions] = useState('')
  const [newTools, setNewTools] = useState<string[]>([])
  const [creating, setCreating] = useState(false)

  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')
  const [editInstructions, setEditInstructions] = useState('')
  const [editTools, setEditTools] = useState<string[]>([])

  const [execAgentId, setExecAgentId] = useState<string | null>(null)
  const [execPrompt, setExecPrompt] = useState('')
  const [execResult, setExecResult] = useState<string | null>(null)
  const [execRunning, setExecRunning] = useState(false)

  const [orchGoal, setOrchGoal] = useState('')
  const [orchContext, setOrchContext] = useState('')
  const [orchRunning, setOrchRunning] = useState(false)
  const [orchPhase, setOrchPhase] = useState<string>('')
  const [orchTasks, setOrchTasks] = useState<OrchestrateTask[]>([])
  const [orchTaskStatuses, setOrchTaskStatuses] = useState<Record<string, string>>({})
  const [orchLevel, setOrchLevel] = useState(0)
  const [orchTotalLevels, setOrchTotalLevels] = useState(0)
  const [orchResponse, setOrchResponse] = useState<string | null>(null)
  const [orchError, setOrchError] = useState<string | null>(null)

  const [runs, setRuns] = useState<AgentRun[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [expandedRun, setExpandedRun] = useState<string | null>(null)

  const fetchAgents = useCallback(async () => {
    setLoading(true)
    try {
      const data = await agentsController.list()
      setAgents(data)
    } catch {
      addToast('Failed to load agents', 'error')
    }
    setLoading(false)
  }, [addToast])

  useEffect(() => { fetchAgents() }, [fetchAgents])

  const fetchRuns = useCallback(async () => {
    setRunsLoading(true)
    try {
      const res = await agentsController.listRuns(20)
      setRuns(res.runs || [])
    } catch {
      addToast('Failed to load run history', 'error')
    }
    setRunsLoading(false)
  }, [addToast])

  useEffect(() => { fetchRuns() }, [fetchRuns])

  const toggleTool = (tool: string, current: string[], setter: (v: string[]) => void) => {
    setter(current.includes(tool) ? current.filter(t => t !== tool) : [...current, tool])
  }

  const handleCreate = async () => {
    if (!newName.trim()) return
    setCreating(true)
    try {
      await agentsController.create({
        name: newName.trim(),
        description: newDesc.trim(),
        instructions: newInstructions.trim(),
        tools: newTools,
      })
      setNewName(''); setNewDesc(''); setNewInstructions(''); setNewTools([])
      addToast('Agent created', 'success')
      await fetchAgents()
    } catch (e) {
      addToast('Failed to create agent', 'error')
    }
    setCreating(false)
  }

  const startEdit = (a: Agent) => {
    setEditingId(a.id)
    setEditName(a.name)
    setEditDesc(a.description)
    setEditInstructions(a.instructions)
    setEditTools(a.tools)
  }

  const cancelEdit = () => {
    setEditingId(null)
    setEditName(''); setEditDesc(''); setEditInstructions(''); setEditTools([])
  }

  const handleUpdate = async (id: string) => {
    try {
      await agentsController.update(id, { name: editName, description: editDesc, instructions: editInstructions, tools: editTools })
      addToast('Agent updated', 'success')
      setEditingId(null)
      await fetchAgents()
    } catch (e) {
      addToast('Update failed', 'error')
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await agentsController.delete(id)
      setAgents(prev => prev.filter(a => a.id !== id))
      addToast('Agent deleted', 'success')
    } catch {
      addToast('Failed to delete agent', 'error')
    }
  }

  const handleExecute = async (id: string) => {
    if (!execPrompt.trim()) return
    setExecRunning(true)
    setExecResult(null)
    try {
      const res = await agentsController.execute(id, execPrompt)
      setExecResult(res.response)
    } catch (e) {
      setExecResult('Execution failed')
    }
    setExecRunning(false)
  }

  const handleOrchestrate = async () => {
    if (!orchGoal.trim()) return
    setOrchRunning(true)
    setOrchPhase('PLAN')
    setOrchTasks([])
    setOrchTaskStatuses({})
    setOrchLevel(0)
    setOrchTotalLevels(0)
    setOrchResponse(null)
    setOrchError(null)

    await agentsController.orchestrate(orchGoal, orchContext, {
      onPlan: (tasks) => {
        setOrchTasks(tasks)
        const statuses: Record<string, string> = {}
        tasks.forEach(t => { statuses[t.id] = 'pending' })
        setOrchTaskStatuses(statuses)
        setOrchPhase('EXECUTE')
      },
      onLevelChange: (level, total) => {
        setOrchLevel(level + 1)
        setOrchTotalLevels(total)
      },
      onTaskStatus: (taskId, status) => {
        setOrchTaskStatuses(prev => ({ ...prev, [taskId]: status }))
      },
      onCompose: () => {
        setOrchPhase('COMPOSE')
      },
      onComplete: (response) => {
        setOrchResponse(response)
        setOrchPhase('COMPLETE')
        setOrchRunning(false)
        addToast('Orchestration complete', 'success')
        fetchRuns()
      },
      onError: (error) => {
        setOrchError(error)
        setOrchPhase('ERROR')
        setOrchRunning(false)
        addToast(error, 'error')
        fetchRuns()
      },
    })
  }

  const toolCount = agents.reduce((acc, a) => acc + a.tools.length, 0)

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Agents" />} />

      <div className="space-y-4">
        {/* Stats */}
        <KpiGrid columns={3}>
          <StatCard label="Total Agents" value={agents.length} />
          <StatCard label="Tool Assignments" value={toolCount} />
          <StatCard label="Available Tools" value={AVAILABLE_TOOLS.length} />
        </KpiGrid>

        {/* Create */}
        <Card>
          <CardHeader><CardTitle className="text-base">New Agent</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Input placeholder="Name" value={newName} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewName(e.target.value)} />
            <Input placeholder="Description (optional)" value={newDesc} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewDesc(e.target.value)} />
            <textarea
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[80px]"
              placeholder="Instructions — how the agent should behave"
              value={newInstructions}
              onChange={e => setNewInstructions(e.target.value)}
            />
            <div className="flex flex-wrap gap-1">
              {AVAILABLE_TOOLS.map(t => (
                <button
                  key={t}
                  onClick={() => toggleTool(t, newTools, setNewTools)}
                  aria-pressed={newTools.includes(t)}
                  className={`rounded-full px-3 py-1 text-xs font-medium border transition-colors ${
                    newTools.includes(t)
                      ? 'bg-primary text-primary-foreground border-primary'
                      : 'bg-background text-muted-foreground border-border hover:border-primary/50'
                  }`}
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

        {/* List */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Agents</CardTitle>
              <Button size="sm" variant="ghost" onClick={fetchAgents} disabled={loading}>
                <IconRefresh className="h-4 w-4 mr-1" />
                Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {loading ? (
              <div className="space-y-2">
                {[1, 2, 3].map(i => <div key={i} className="h-16 animate-pulse rounded-lg bg-muted" />)}
              </div>
            ) : agents.length === 0 ? (
              <EmptyCard message="No agents yet" action={null} />
            ) : (
              agents.map(agent => (
                <div key={agent.id} className={`rounded-lg border p-3 space-y-2 transition-colors ${execAgentId === agent.id ? 'bg-primary/[0.08] border-primary/40' : 'border-border/60 hover:bg-muted/50'}`}>
                  {editingId === agent.id ? (
                    <div className="space-y-2">
                      <Input value={editName} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditName(e.target.value)} placeholder="Name" />
                      <Input value={editDesc} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditDesc(e.target.value)} placeholder="Description" />
                      <textarea
                        className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[60px]"
                        value={editInstructions}
                        onChange={e => setEditInstructions(e.target.value)}
                      />
                      <div className="flex flex-wrap gap-1">
                        {AVAILABLE_TOOLS.map(t => (
                          <button
                            key={t}
                            onClick={() => toggleTool(t, editTools, setEditTools)}
                            aria-pressed={editTools.includes(t)}
                            className={`rounded-full px-3 py-1 text-xs font-medium border transition-colors ${
                              editTools.includes(t)
                                ? 'bg-primary text-primary-foreground border-primary'
                                : 'bg-background text-muted-foreground border-border hover:border-primary/50'
                            }`}
                          >
                            {t.replace(/_/g, ' ')}
                          </button>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => handleUpdate(agent.id)}>Save</Button>
                        <Button size="sm" variant="ghost" onClick={cancelEdit}>Cancel</Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-start justify-between">
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium">{agent.name}</p>
                          {agent.description && (
                            <p className="text-xs text-muted-foreground mt-0.5">{agent.description}</p>
                          )}
                        </div>
                        <div className="flex gap-1 shrink-0 ml-2">
                          <Button size="sm" variant="ghost" onClick={() => startEdit(agent)}>Edit</Button>
                           <Button size="sm" variant="destructive" onClick={() => handleDelete(agent.id)} aria-label={`Delete ${agent.name}`}>
                            <IconTrash className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </div>
                      {agent.tools.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {agent.tools.map(t => (
                            <span key={t} className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                              {t.replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                      )}
                      {agent.instructions && (
                        <p className="text-[10px] text-muted-foreground/70 line-clamp-2">{agent.instructions}</p>
                      )}
                      {/* Inline execute */}
                      <div className="pt-1">
                        {execAgentId === agent.id ? (
                          <div className="space-y-2">
                            <Input
                              placeholder="What should this agent do?"
                              value={execPrompt}
                              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setExecPrompt(e.target.value)}
                              onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleExecute(agent.id) } }}
                            />
                            <div className="flex gap-2">
                              <Button size="sm" onClick={() => handleExecute(agent.id)} disabled={execRunning || !execPrompt.trim()}>
                                {execRunning ? 'Running...' : 'Execute'}
                              </Button>
                              <Button size="sm" variant="ghost" onClick={() => { setExecAgentId(null); setExecResult(null); setExecPrompt('') }}>
                                Close
                              </Button>
                            </div>
                            {execResult && (
                              <div className="rounded-lg bg-muted p-3">
                                <p className="text-xs font-medium text-muted-foreground mb-1">Response</p>
                                <p className="text-sm whitespace-pre-wrap">{execResult}</p>
                              </div>
                            )}
                          </div>
                        ) : (
                          <Button size="sm" variant="outline" onClick={() => { setExecAgentId(agent.id); setExecPrompt(''); setExecResult(null) }}>
                            Run
                          </Button>
                        )}
                      </div>
                    </>
                  )}
                </div>
              ))
            )}
          </CardContent>
        </Card>

        {/* Orchestration */}
        <Card>
          <CardHeader><CardTitle className="text-base">Multi-Agent Orchestration</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Decompose a goal into subtasks and execute them across multiple agents in parallel.
            </p>
            <Input
              placeholder="Goal — e.g. research transformers and write a summary"
              value={orchGoal}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setOrchGoal(e.target.value)}
              onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter' && !e.shiftKey && orchGoal.trim() && !orchRunning) { e.preventDefault(); handleOrchestrate() } }}
            />
            <textarea
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[60px]"
              placeholder="Additional context (optional)"
              value={orchContext}
              onChange={e => setOrchContext(e.target.value)}
            />
            <div className="flex gap-2">
              <Button onClick={handleOrchestrate} disabled={orchRunning || !orchGoal.trim()}>
                {orchRunning ? 'Orchestrating...' : 'Orchestrate'}
              </Button>
              {(orchResponse || orchError) && (
                <Button size="sm" variant="ghost" onClick={() => { setOrchResponse(null); setOrchError(null); setOrchPhase(''); setOrchTasks([]); setOrchTaskStatuses({}) }}>
                  Clear
                </Button>
              )}
            </div>

            {/* Phase indicator */}
            {orchPhase && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className={`inline-block h-2 w-2 rounded-full ${
                  orchPhase === 'COMPLETE' ? 'bg-success' :
                  orchPhase === 'ERROR' ? 'bg-destructive' :
                  'bg-warning animate-pulse'
                }`} />
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
                <p className="text-xs font-medium text-muted-foreground">Tasks ({orchTasks.length})</p>
                {orchTasks.map(task => {
                  const status = orchTaskStatuses[task.id] || 'pending'
                  return (
                    <div key={task.id} className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm">
                      <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${
                        status === 'completed' ? 'bg-success' :
                        status === 'in_progress' ? 'bg-warning animate-pulse' :
                        status === 'failed' ? 'bg-destructive' :
                        'bg-muted-foreground/30'
                      }`} />
                      <span className="font-medium text-xs min-w-[64px] text-muted-foreground">{task.agent}</span>
                      <span className="flex-1 truncate">{task.description}</span>
                      {task.depends_on && task.depends_on.length > 0 && (
                        <span className="text-[10px] text-muted-foreground/50">after: {task.depends_on.join(', ')}</span>
                      )}
                    </div>
                  )
                })}
              </div>
            )}

            {/* Final response */}
            {orchResponse && (
              <div className="rounded-lg border bg-muted/30 p-3">
                <p className="text-xs font-medium text-muted-foreground mb-1">Result</p>
                <p className="text-sm whitespace-pre-wrap">{orchResponse}</p>
              </div>
            )}

            {/* Error */}
            {orchError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                {orchError}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Run history */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Run History</CardTitle>
              <Button size="sm" variant="ghost" onClick={fetchRuns} disabled={runsLoading}>
                <IconRefresh className="h-4 w-4 mr-1" />
                Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {runsLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map(i => <div key={i} className="h-14 animate-pulse rounded-lg bg-muted" />)}
              </div>
            ) : runs.length === 0 ? (
              <EmptyCard message="No runs yet — orchestrate a goal to see history here" action={null} />
            ) : (
              runs.map(run => {
                const expanded = expandedRun === run.id
                const statusColor = run.status === 'completed' ? 'bg-success' : run.status === 'failed' ? 'bg-destructive' : 'bg-warning animate-pulse'
                return (
                  <div key={run.id} className={`rounded-lg border border-border/60 transition-colors ${expanded ? 'border-primary/40' : 'hover:bg-muted/50'}`}>
                    <button
                      className="w-full flex items-center gap-2 px-3 py-2 text-left"
                      onClick={() => setExpandedRun(expanded ? null : run.id)}
                      aria-expanded={expanded}
                    >
                      <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${statusColor}`} />
                      <span className="flex-1 truncate text-sm">{run.goal}</span>
                      <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                        {run.completed_count}/{run.completed_count + run.failed_count} tasks
                      </span>
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {run.started_at ? new Date(run.started_at).toLocaleString() : ''}
                      </span>
                    </button>
                    {expanded && (
                      <div className="border-t border-border/60 px-3 py-2 space-y-3">
                        {run.error && (
                          <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                            {run.error}
                          </div>
                        )}
                        {run.tasks.length > 0 && (
                          <div className="space-y-1.5">
                            <p className="text-xs font-medium text-muted-foreground">Tasks ({run.tasks.length})</p>
                            {run.tasks.map(task => (
                              <div key={task.id} className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm">
                                <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${
                                  task.status === 'completed' ? 'bg-success' :
                                  task.status === 'failed' ? 'bg-destructive' :
                                  'bg-muted-foreground/30'
                                }`} />
                                <span className="font-medium text-xs min-w-[64px] text-muted-foreground">{task.agent}</span>
                                <span className="flex-1 truncate">{task.description}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {run.response && (
                          <div className="rounded-lg border bg-muted/30 p-3">
                            <p className="text-xs font-medium text-muted-foreground mb-1">Result</p>
                            <p className="text-sm whitespace-pre-wrap line-clamp-4">{run.response}</p>
                          </div>
                        )}
                        {run.logs.length > 0 && (
                          <div className="rounded-lg bg-muted/50 p-3">
                            <p className="text-xs font-medium text-muted-foreground mb-1">Logs</p>
                            <pre className="text-[10px] text-muted-foreground whitespace-pre-wrap max-h-40 overflow-y-auto">
                              {run.logs.join('\n')}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
