'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useEffect, useRef, useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { SectionErrorBoundary } from '@/components/SectionErrorBoundary'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Checkbox } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { EmptyCard, KpiGrid, StatCard, IconRefresh, IconCopy, cn, Skeleton } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { IconPlus, IconTrash, IconClock } from '@/components/icons/NavIcons'
import { agentsController, type Agent, type OrchestrateTask, type AgentRun } from '@/lib/agents-controller'
import { useToastStore } from '@/lib/toast-store'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString } from '@/lib/format-bytes'
import { formatElapsed } from '@/lib/formatDuration'
import { agentSchema, agentExecuteSchema, orchestrateSchema } from '@/lib/validation-schemas'
import { logger } from '@/lib/dev-log'

const AVAILABLE_TOOLS = ['code_execution', 'file_read', 'file_search', 'web_search', 'knowledge_retrieval', 'image_analysis', 'data_analysis', 'citation']

const AGENT_TEMPLATES = [
  { name: 'Researcher', desc: 'Finds information from the web and documents', instructions: 'You are a thorough researcher. Search for accurate, up-to-date information and present findings clearly with sources.', tools: ['web_search', 'knowledge_retrieval'] },
  { name: 'Coder', desc: 'Writes and debugs code', instructions: 'You are an expert programmer. Write clean, efficient code. Explain your approach. Debug errors methodically.', tools: ['code_execution', 'file_read'] },
  { name: 'Analyst', desc: 'Analyzes data and generates insights', instructions: 'You are a data analyst. Examine datasets, identify patterns, compute statistics, and present clear visualizations and insights.', tools: ['data_analysis', 'file_read'] },
  { name: 'Writer', desc: 'Creates structured written content', instructions: 'You are a skilled writer. Produce clear, well-organized content. Adapt tone to the audience. Ensure accuracy.', tools: ['knowledge_retrieval'] },
  { name: 'Vision Assistant', desc: 'Analyzes images and visual data', instructions: 'You are a vision expert. Analyze images carefully, describe what you see, identify objects, text, and patterns.', tools: ['image_analysis'] },
  { name: 'Full Stack', desc: 'Handles all aspects of a task', instructions: 'You are a versatile full-stack assistant. Use whatever tools are needed to complete the task: search, code, analyze, or write.', tools: ['web_search', 'code_execution', 'file_read', 'knowledge_retrieval', 'data_analysis', 'file_search'] },
]

type RunStatus = 'completed' | 'failed' | 'running' | string

function statusDotColor(status: RunStatus): string {
  if (status === 'completed') return 'bg-success'
  if (status === 'failed') return 'bg-destructive'
  return 'bg-warning animate-pulse'
}

function statusTextColor(status: RunStatus): string {
  if (status === 'completed') return 'text-success'
  if (status === 'failed') return 'text-destructive'
  return 'text-warning'
}

function taskDotColor(status: RunStatus): string {
  if (status === 'completed') return 'bg-success'
  if (status === 'failed') return 'bg-destructive'
  return 'bg-muted-foreground/30'
}

function taskBadgeStyle(status: RunStatus): string {
  if (status === 'completed') return 'bg-success/15 text-success'
  if (status === 'failed') return 'bg-destructive/15 text-destructive'
  return 'bg-muted text-muted-foreground'
}

function getAgentTaskStats(tasks: OrchestrateTask[]): Record<string, { completed: number; failed: number; total: number }> {
  const stats: Record<string, { completed: number; failed: number; total: number }> = {}
  for (const t of tasks) {
    if (!stats[t.agent]) stats[t.agent] = { completed: 0, failed: 0, total: 0 }
    stats[t.agent].total++
    if (t.status === 'completed') stats[t.agent].completed++
    else if (t.status === 'failed') stats[t.agent].failed++
  }
  return stats
}

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
  const [execToolsUsed, setExecToolsUsed] = useState<Array<{ tool: string; result: unknown }>>([])
  const [pendingDelete, setPendingDelete] = useState<Agent | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [execRunning, setExecRunning] = useState(false)
  const [agentSearch, setAgentSearch] = useState('')

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
  const [orchAgentIds, setOrchAgentIds] = useState<string[]>([])

  const [runs, setRuns] = useState<AgentRun[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [expandedRun, setExpandedRun] = useState<string | null>(null)
  const [runStatusFilter, setRunStatusFilter] = useState<string | null>(null)
  const [runAgentFilter, setRunAgentFilter] = useState<string | null>(null)
  const [runViewMode, setRunViewMode] = useState<'list' | 'timeline'>('list')

  const [createErrors, setCreateErrors] = useState<{ name?: string; description?: string; instructions?: string }>({})
  const [editErrors, setEditErrors] = useState<{ name?: string; description?: string; instructions?: string }>({})
  const [execErrors, setExecErrors] = useState<{ prompt?: string }>({})
  const [orchErrors, setOrchErrors] = useState<{ goal?: string; context?: string }>({})

  const fetchAgents = useCallback(async () => {
    setLoading(true)
    try {
      const data = await agentsController.list()
      setAgents(data)
    } catch {
      addToast('Could not load agents', 'error')
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
      addToast('Could not load run history', 'error')
    }
    setRunsLoading(false)
  }, [addToast])

  useEffect(() => { fetchRuns() }, [fetchRuns])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return
      if (e.key === 'n' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        document.querySelector<HTMLInputElement>('[placeholder="Name"]')?.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const importInputRef = useRef<HTMLInputElement>(null)

  const handleExport = () => {
    if (agents.length === 0) return addToast('No agents to export', 'error')
    const data = agents.map(a => ({ name: a.name, description: a.description, instructions: a.instructions, tools: a.tools }))
    downloadJson(data, `agents-${todayDateString()}.json`)
    addToast(`Exported ${agents.length} agents`, 'success')
  }

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = async (ev) => {
      try {
        const imported = JSON.parse(ev.target?.result as string)
        if (!Array.isArray(imported)) throw new Error('Invalid format')
        let count = 0
        for (const item of imported) {
          if (!item.name) continue
          try {
            await agentsController.create({
              name: item.name,
              description: item.description || '',
              instructions: item.instructions || '',
              tools: Array.isArray(item.tools) ? item.tools : [],
            })
            count++
          } catch { /* skip individual failures */ }
        }
        addToast(`Imported ${count} agents`, 'success')
        await fetchAgents()
      } catch {
        addToast('Invalid agent file', 'error')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  const toggleTool = (tool: string, current: string[], setter: (v: string[]) => void) => {
    setter(current.includes(tool) ? current.filter(t => t !== tool) : [...current, tool])
  }

  const handleCreate = async () => {
    const result = agentSchema.safeParse({ name: newName, description: newDesc, instructions: newInstructions })
    if (!result.success) {
      const fieldErrors: { name?: string; description?: string; instructions?: string } = {}
      result.error.issues.forEach(issue => {
        const field = issue.path[0] as string
        if (field === 'name') fieldErrors.name = issue.message
        if (field === 'description') fieldErrors.description = issue.message
        if (field === 'instructions') fieldErrors.instructions = issue.message
      })
      setCreateErrors(fieldErrors)
      return
    }
    setCreateErrors({})
    setCreating(true)
    const trimmedName = newName.trim()
    const trimmedDesc = newDesc.trim()
    const trimmedInstructions = newInstructions.trim()
    const tempId = `temp-${Date.now()}`
    const optimistic: Agent = {
      id: tempId,
      name: trimmedName,
      description: trimmedDesc,
      instructions: trimmedInstructions,
      tools: newTools,
      avatar: '',
    }
    setAgents(prev => [optimistic, ...prev])
    setNewName(''); setNewDesc(''); setNewInstructions(''); setNewTools([])
    try {
      const created = await agentsController.create({
        name: trimmedName,
        description: trimmedDesc,
        instructions: trimmedInstructions,
        tools: newTools,
      })
      setAgents(prev => prev.map(a => a.id === tempId ? created : a))
      addToast('Agent created', 'success')
    } catch {
      setAgents(prev => prev.filter(a => a.id !== tempId))
      addToast('Could not create agent', 'error')
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
    const result = agentSchema.safeParse({ name: editName, description: editDesc, instructions: editInstructions })
    if (!result.success) {
      const fieldErrors: { name?: string; description?: string; instructions?: string } = {}
      result.error.issues.forEach(issue => {
        const field = issue.path[0] as string
        if (field === 'name') fieldErrors.name = issue.message
        if (field === 'description') fieldErrors.description = issue.message
        if (field === 'instructions') fieldErrors.instructions = issue.message
      })
      setEditErrors(fieldErrors)
      return
    }
    setEditErrors({})
    const oldAgent = agents.find(a => a.id === id)
    const trimmedName = editName.trim()
    const trimmedDesc = editDesc.trim()
    const trimmedInstructions = editInstructions.trim()
    setAgents(prev => prev.map(a => a.id === id ? { ...a, name: trimmedName, description: trimmedDesc, instructions: trimmedInstructions, tools: editTools } : a))
    setEditingId(null)
    try {
      await agentsController.update(id, { name: trimmedName, description: trimmedDesc, instructions: trimmedInstructions, tools: editTools })
      addToast('Agent updated', 'success')
    } catch {
      if (oldAgent) setAgents(prev => prev.map(a => a.id === id ? oldAgent : a))
      addToast('Could not update', 'error')
    }
  }

  const handleDelete = async (id: string) => {
    const deleted = agents.find(a => a.id === id)
    if (!deleted) return
    setAgents(prev => prev.filter(a => a.id !== id))
    try {
      await agentsController.delete(id)
      addToast('Agent deleted', 'success', undefined, () => {
        setAgents(prev => [deleted, ...prev])
        agentsController.create({ name: deleted.name, description: deleted.description, instructions: deleted.instructions, tools: deleted.tools }).catch((e) => {
          logger.warning('Could not restore agent', { exception: String(e) })
          addToast('Could not restore agent', 'error')
          fetchAgents()
        })
      })
    } catch {
      setAgents(prev => [deleted, ...prev])
      addToast('Could not delete agent', 'error')
    }
  }

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const selectAll = () => {
    if (selectedIds.size === filteredAgents.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filteredAgents.map(a => a.id)))
    }
  }

  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) return
    const deletedIds = new Set(selectedIds)
    const deletedAgents = agents.filter(a => deletedIds.has(a.id))
    setAgents(prev => prev.filter(a => !deletedIds.has(a.id)))
    setSelectedIds(new Set())
    try {
      await Promise.all(Array.from(deletedIds).map(id => agentsController.delete(id)))
      addToast(`Deleted ${deletedIds.size} agents`, 'success')
    } catch {
      setAgents(prev => [...deletedAgents, ...prev])
      addToast('Could not delete some agents', 'error')
    }
  }

  const handleBulkExport = () => {
    const selected = agents.filter(a => selectedIds.has(a.id))
    downloadJson(selected, `agents-export-${todayDateString()}.json`)
    addToast(`Exported ${selected.length} agents`, 'success')
    setSelectedIds(new Set())
  }

  const handleExecute = async (id: string) => {
    const result = agentExecuteSchema.safeParse({ prompt: execPrompt })
    if (!result.success) {
      setExecErrors({ prompt: result.error.issues[0]?.message })
      return
    }
    setExecErrors({})
    setExecRunning(true)
    setExecResult(null)
    setExecToolsUsed([])
    try {
      const res = await agentsController.execute(id, execPrompt)
      setExecResult(res.response)
      setExecToolsUsed(res.tools_used || [])
    } catch {
      setExecResult('Could not execute agent')
    }
    setExecRunning(false)
  }

  const handleOrchestrate = async () => {
    const result = orchestrateSchema.safeParse({ goal: orchGoal, context: orchContext })
    if (!result.success) {
      const fieldErrors: { goal?: string; context?: string } = {}
      result.error.issues.forEach(issue => {
        const field = issue.path[0] as string
        if (field === 'goal') fieldErrors.goal = issue.message
        if (field === 'context') fieldErrors.context = issue.message
      })
      setOrchErrors(fieldErrors)
      return
    }
    setOrchErrors({})
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
    }, undefined, orchAgentIds.length > 0 ? orchAgentIds : undefined)
  }

  const toolCount = agents.reduce((acc, a) => acc + (a.tools?.length ?? 0), 0)

  const filteredAgents = agentSearch
    ? agents.filter(a => a.name.toLowerCase().includes(agentSearch.toLowerCase()) || (a.description?.toLowerCase() ?? '').includes(agentSearch.toLowerCase()))
    : agents

  const handleClone = async (agent: Agent) => {
    try {
      await agentsController.create({
        name: `${agent.name} (copy)`,
        description: agent.description,
        instructions: agent.instructions,
        tools: [...agent.tools],
      })
      addToast(`Cloned "${agent.name}"`, 'success')
      await fetchAgents()
    } catch {
      addToast('Could not clone agent', 'error')
    }
  }

  const headerRight = (
    <div className="flex items-center gap-2">
      <input ref={importInputRef} type="file" accept=".json" className="hidden" onChange={handleImport} />
      <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={handleExport} disabled={agents.length === 0}>
        Export
      </Button>
      <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={() => importInputRef.current?.click()}>
        Import
      </Button>
    </div>
  )

  return (
    <PageContainer
      title="Agents"
      headerRight={headerRight}
    >
        {/* Stats */}
        <KpiGrid columns={3}>
          <StatCard label="Total Agents" value={agents.length} />
          <StatCard label="Tool Assignments" value={toolCount} />
          <StatCard label="Available Tools" value={AVAILABLE_TOOLS.length} />
        </KpiGrid>

        {/* Create */}
        <SectionErrorBoundary sectionName="Agent creator">
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5"><CardTitle className="text-[11px] font-medium">New Agent</CardTitle></CardHeader>
          <CardContent className="px-2.5 pb-2.5 space-y-3">
            <div>
              <p className="text-[10px] font-medium text-muted-foreground mb-2">Start from a template</p>
              <div className="flex flex-wrap gap-1.5">
                {AGENT_TEMPLATES.map(t => (
                  <button
                    key={t.name}
                    type="button"
                    onClick={() => { setNewName(t.name); setNewDesc(t.desc); setNewInstructions(t.instructions); setNewTools([...t.tools]) }}
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
                  if (createErrors.name) setCreateErrors(prev => ({ ...prev, name: undefined }))
                }}
                className={createErrors.name ? 'border-destructive ring-destructive/20' : ''}
                aria-invalid={!!createErrors.name}
                aria-describedby={createErrors.name ? 'create-name-error' : undefined}
              />
              {createErrors.name && <p id="create-name-error" className="text-[10px] text-destructive mt-1" role="alert">{createErrors.name}</p>}
            </div>
            <Input placeholder="Description (optional)" value={newDesc} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewDesc(e.target.value)} />
            <textarea
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-[11px] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[80px]"
              placeholder="Instructions — how the agent should behave"
              value={newInstructions}
              onChange={e => setNewInstructions(e.target.value)}
              aria-label="Agent instructions"
            />
            <div className="flex flex-wrap gap-1">
              {AVAILABLE_TOOLS.map(t => (
                <button
                  key={t}
                  type="button"
                  onClick={() => toggleTool(t, newTools, setNewTools)}
                  aria-pressed={newTools.includes(t)}
                  className={cn('rounded-full px-3 py-1 text-[10px] font-medium border transition-colors', newTools.includes(t) ? 'bg-primary text-primary-foreground border-primary' : 'bg-background text-muted-foreground border-border hover:border-primary/50')}
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

        {/* List */}
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <div className="flex items-center justify-between">
              <CardTitle className="text-[11px] font-medium">Agents</CardTitle>
              <Button size="sm" variant="ghost" onClick={fetchAgents} disabled={loading} className="h-6 text-[10px]">
                <IconRefresh className="h-4 w-4 mr-1" />
                Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5 space-y-2">
            {loading ? (
              <div className="space-y-2">
                {[1, 2, 3].map(i => (
                  <div key={i} className="flex items-center gap-1.5 p-2.5 rounded-lg border border-border/40">
                    <Skeleton className="h-6 w-6 rounded shrink-0 bg-muted/20" />
                    <div className="flex-1 space-y-1">
                      <Skeleton className="h-3.5 w-24 bg-muted/20 rounded-lg" />
                      <Skeleton className="h-3 w-40 bg-muted/20 rounded-lg" />
                    </div>
                    <Skeleton className="h-6 w-14 rounded-full bg-muted/20" />
                  </div>
                ))}
              </div>
            ) : agents.length === 0 ? (
              <EmptyCard
                message="No agents yet"
                description="Create your first agent above. Use templates for quick setup."
                icon={<IconPlus className="h-5 w-5" />}
                action={null}
              />
            ) : (
              <>
                {agents.length > 2 && (
                  <input
                    type="text"
                    aria-label="Search agents"
                    value={agentSearch}
                    onChange={e => setAgentSearch(e.target.value)}
                    placeholder="Search agents..."
                    className="h-7 w-full max-w-xs rounded-md border border-border/60 bg-background px-2.5 text-[11px] placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/30"
                  />
                )}
                {filteredAgents.length > 0 && (
                  <div className="flex items-center gap-2 text-[10px]">
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <Checkbox
                        checked={selectedIds.size === filteredAgents.length && filteredAgents.length > 0}
                        onCheckedChange={selectAll}
                        aria-label="Select all agents"
                        className="rounded border-border"
                      />
                      <span className="text-muted-foreground">Select all</span>
                    </label>
                    {selectedIds.size > 0 && (
                      <>
                        <span className="text-muted-foreground">·</span>
                        <span className="text-muted-foreground">{selectedIds.size} selected</span>
                        <Button size="sm" variant="ghost" onClick={handleBulkExport} className="h-6 text-[10px]">
                          Export
                        </Button>
                        <Button size="sm" variant="ghost" onClick={handleBulkDelete} className="h-6 text-[10px] text-destructive hover:text-destructive">
                          Delete
                        </Button>
                      </>
                    )}
                  </div>
                )}
                {filteredAgents.length === 0 ? (
                  <div className="text-center py-4">
                    <p className="text-[10px] text-muted-foreground/60">No agents matching &quot;{agentSearch}&quot;</p>
                    <Button size="sm" variant="ghost" className="h-6 text-[10px] mt-2" onClick={() => setAgentSearch('')}>
                      Clear search
                    </Button>
                  </div>
                ) : (
                  filteredAgents.map(agent => (
                <div key={agent.id} className={cn('rounded-lg border space-y-2 transition-colors', execAgentId === agent.id ? 'p-2.5 bg-primary/[0.08] border-primary/40' : 'border-border/40 p-2.5 hover:bg-muted/20')}>
                  {editingId === agent.id ? (
                    <div className="space-y-2">
                      <div>
                        <Input
                          id="edit-agent-name"
                          value={editName}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            setEditName(e.target.value)
                            if (editErrors.name) setEditErrors(prev => ({ ...prev, name: undefined }))
                          }}
                          placeholder="Name"
                          aria-label="Agent name"
                          className={editErrors.name ? 'border-destructive ring-destructive/20' : ''}
                          aria-invalid={!!editErrors.name}
                          aria-describedby={editErrors.name ? 'edit-name-error' : undefined}
                        />
                        {editErrors.name && <p id="edit-name-error" className="text-[10px] text-destructive mt-1" role="alert">{editErrors.name}</p>}
                      </div>
                      <Input value={editDesc} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditDesc(e.target.value)} placeholder="Description" aria-label="Agent description" />
                      <textarea
                        className="w-full rounded-lg border border-input bg-background px-3 py-2 text-[11px] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[60px]"
                        value={editInstructions}
                        onChange={e => setEditInstructions(e.target.value)}
                        aria-label="Edit agent instructions"
                      />
                      <div className="flex flex-wrap gap-1">
                        {AVAILABLE_TOOLS.map(t => (
                          <button
                            key={t}
                            type="button"
                            onClick={() => toggleTool(t, editTools, setEditTools)}
                            aria-pressed={editTools.includes(t)}
                            className={cn('rounded-full px-3 py-1 text-[10px] font-medium border transition-colors', editTools.includes(t) ? 'bg-primary text-primary-foreground border-primary' : 'bg-background text-muted-foreground border-border hover:border-primary/50')}
                          >
                            {t.replace(/_/g, ' ')}
                          </button>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => handleUpdate(agent.id)} className="h-7 text-[11px]">Save</Button>
                        <Button size="sm" variant="ghost" onClick={cancelEdit} className="h-6 text-[10px]">Cancel</Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-2 min-w-0 flex-1">
                          <Checkbox
                            checked={selectedIds.has(agent.id)}
                            onCheckedChange={() => toggleSelect(agent.id)}
                            aria-label={`Select agent ${agent.name}`}
                            className="mt-1 rounded border-border"
                          />
                          <div className="min-w-0 flex-1">
                            <p className="text-[11px] font-medium">{agent.name}</p>
                            {agent.description && (
                              <p className="text-[10px] text-muted-foreground mt-0.5">{agent.description}</p>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-1 shrink-0 ml-2">
                          <Button size="sm" variant="ghost" onClick={() => handleClone(agent)} aria-label={`Clone ${agent.name}`} className="h-6 text-[10px]">
                            <IconCopy className="h-4 w-4" />
                          </Button>
                          <Button size="sm" variant="ghost" onClick={() => startEdit(agent)} className="h-6 text-[10px]">Edit</Button>
                           <Button size="sm" variant="destructive" onClick={() => setPendingDelete(agent)} aria-label={`Delete ${agent.name}`} className="h-6 text-[10px]">
                            <IconTrash className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      {agent.tools.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {agent.tools.map(t => (
                            <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
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
                            <div>
                              <Input
                                id="exec-prompt"
                                placeholder="What should this agent do?"
                                value={execPrompt}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                                  setExecPrompt(e.target.value)
                                  if (execErrors.prompt) setExecErrors({})
                                }}
                                onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleExecute(agent.id) } }}
                                className={execErrors.prompt ? 'border-destructive ring-destructive/20' : ''}
                                aria-invalid={!!execErrors.prompt}
                                aria-describedby={execErrors.prompt ? 'exec-prompt-error' : undefined}
                              />
                              {execErrors.prompt && <p id="exec-prompt-error" className="text-[10px] text-destructive mt-1" role="alert">{execErrors.prompt}</p>}
                            </div>
                            <div className="flex gap-2">
                              <Button size="sm" onClick={() => handleExecute(agent.id)} disabled={execRunning || !execPrompt.trim()}>
                                {execRunning ? 'Running...' : 'Execute'}
                              </Button>
                              <Button size="sm" variant="ghost" onClick={() => { setExecAgentId(null); setExecResult(null); setExecPrompt('') }} className="h-6 text-[10px]">
                                Close
                              </Button>
                            </div>
                            {execResult && (
                              <div className="rounded-lg bg-muted p-3">
                                {execToolsUsed.length > 0 && (
                                  <div className="mb-2 flex flex-wrap gap-1">
                                    {execToolsUsed.map((t, i) => (
                                      <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                                        {t.tool.replace(/_/g, ' ')}
                                      </span>
                                    ))}
                                  </div>
                                )}
                                <p className="text-[10px] font-medium text-muted-foreground mb-1">Response</p>
                                <p className="text-[11px] whitespace-pre-wrap">{execResult}</p>
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
              </>
            )}
          </CardContent>
        </Card>

        {/* Orchestration */}
        <SectionErrorBoundary sectionName="Orchestrator panel">
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5"><CardTitle className="text-[11px] font-medium">Multi-Agent Orchestration</CardTitle></CardHeader>
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
                  if (orchErrors.goal) setOrchErrors(prev => ({ ...prev, goal: undefined }))
                }}
                onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter' && !e.shiftKey && orchGoal.trim() && !orchRunning) { e.preventDefault(); handleOrchestrate() } }}
                className={orchErrors.goal ? 'border-destructive ring-destructive/20' : ''}
                aria-invalid={!!orchErrors.goal}
                aria-describedby={orchErrors.goal ? 'orch-goal-error' : undefined}
              />
              {orchErrors.goal && <p id="orch-goal-error" className="text-[10px] text-destructive mt-1" role="alert">{orchErrors.goal}</p>}
            </div>
            <textarea
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-[11px] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[60px]"
              placeholder="Additional context (optional)"
              value={orchContext}
              onChange={e => setOrchContext(e.target.value)}
              aria-label="Orchestration context"
            />
            {agents.length > 0 && (
              <div>
                <p className="text-[10px] font-medium text-muted-foreground mb-1.5">Agents (optional — leave empty for all)</p>
                <div className="flex flex-wrap gap-1.5">
                  {agents.map(a => (
                    <button
                      key={a.id}
                      type="button"
                      onClick={() => {
                        setOrchAgentIds(prev =>
                          prev.includes(a.id) ? prev.filter(id => id !== a.id) : [...prev, a.id]
                        )
                      }}
                      className={cn('rounded-full px-2.5 py-1 text-[10px] font-medium border transition-colors', orchAgentIds.includes(a.id) ? 'bg-primary text-primary-foreground border-primary' : 'bg-background text-muted-foreground border-border hover:border-primary/50')}
                    >
                      {a.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="flex gap-2">
              <Button onClick={handleOrchestrate} disabled={orchRunning || !orchGoal.trim()} className="h-7 text-[11px]">
                {orchRunning ? 'Orchestrating...' : 'Orchestrate'}
              </Button>
              {(orchResponse || orchError) && (
                <Button size="sm" variant="ghost" onClick={() => { setOrchResponse(null); setOrchError(null); setOrchPhase(''); setOrchTasks([]); setOrchTaskStatuses({}) }} className="h-6 text-[10px]">
                  Clear
                </Button>
              )}
            </div>

            {/* Phase indicator */}
            {orchPhase && (
              <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                <span className={cn('inline-block h-2 w-2 rounded-full', orchPhase === 'COMPLETE' ? 'bg-success' : orchPhase === 'ERROR' ? 'bg-destructive' : 'bg-warning animate-pulse')} />
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
                <p className="text-[10px] font-medium text-muted-foreground">Tasks ({orchTasks.length})</p>
                {orchTasks.map(task => {
                  const status = orchTaskStatuses[task.id] || 'pending'
                  return (
                    <div key={task.id} className="flex items-center gap-2 rounded-lg border border-border/40 px-2.5 py-2 text-[11px]">
                      <span className={cn('inline-block h-2 w-2 shrink-0 rounded-full', status === 'completed' ? 'bg-success' : status === 'in_progress' ? 'bg-warning animate-pulse' : status === 'failed' ? 'bg-destructive' : 'bg-muted-foreground/30')} />
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

            {/* Final response */}
            {orchResponse && (
              <div className="rounded-lg border border-border/40 bg-muted/30 p-3">
                <p className="text-[10px] font-medium text-muted-foreground mb-1">Result</p>
                <p className="text-[11px] whitespace-pre-wrap">{orchResponse}</p>
              </div>
            )}

            {/* Error */}
            {orchError && (
              <StatusBanner variant="error" message={orchError} dismissible={false} />
            )}
          </CardContent>
        </Card>
        </SectionErrorBoundary>

        {/* Run history */}
        <SectionErrorBoundary sectionName="Run history viewer">
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <div className="flex items-center justify-between">
              <CardTitle className="text-[11px] font-medium">Run History</CardTitle>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setRunViewMode('list')}
                  className={cn('text-[10px] px-2 py-1 rounded transition-colors', runViewMode === 'list' ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:bg-muted/80')}
                >
                  List
                </button>
                <button
                  type="button"
                  onClick={() => setRunViewMode('timeline')}
                  className={cn('text-[10px] px-2 py-1 rounded transition-colors', runViewMode === 'timeline' ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:bg-muted/80')}
                >
                  Timeline
                </button>
                <Button size="sm" variant="ghost" onClick={fetchRuns} disabled={runsLoading} className="h-6 text-[10px]">
                  <IconRefresh className="h-4 w-4 mr-1" />
                  Refresh
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5 space-y-2">
            {runsLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map(i => (
                  <div key={i} className="flex items-center gap-1.5 p-2.5 rounded-lg border border-border/40">
                    <Skeleton className="h-6 w-6 rounded shrink-0 bg-muted/20" />
                    <div className="flex-1 space-y-1">
                      <Skeleton className="h-3.5 w-32 bg-muted/20 rounded-lg" />
                      <Skeleton className="h-3 w-20 bg-muted/20 rounded-lg" />
                    </div>
                    <Skeleton className="h-5 w-16 rounded-full bg-muted/20" />
                  </div>
                ))}
              </div>
            ) : runs.length === 0 ? (
              <EmptyCard message="No runs yet — orchestrate a goal to see history here" action={null} />
            ) : (
              <div className="space-y-3">
                {runs.length > 2 && (
                  <div className="space-y-2 pb-2">
                    <div className="flex items-center gap-1.5">
                      <div className="flex gap-1">
                        {[null, 'completed', 'failed', 'running'].map(s => (
                          <button
                            key={s ?? 'all'}
                            type="button"
                            onClick={() => setRunStatusFilter(s)}
                            className={cn('text-[10px] px-2 py-1 rounded-full border transition-colors', runStatusFilter === s ? 'bg-primary/15 text-primary border-primary/30' : 'border-border/40 text-muted-foreground hover:bg-muted/80')}
                          >
                            {s === null ? 'All' : s}
                          </button>
                        ))}
                      </div>
                      <div className="flex-1" />
                      <div className="flex gap-1.5 text-[10px] text-muted-foreground/60">
                        <span>{runs.filter(r => r.status === 'completed').length} completed</span>
                        <span>{runs.filter(r => r.status === 'failed').length} failed</span>
                        <span>{runs.reduce((s, r) => s + r.completed_count + r.failed_count, 0)} total tasks</span>
                      </div>
                    </div>
                    {(() => {
                      const agentNames = new Set<string>()
                      runs.forEach(r => r.tasks.forEach(t => { if (t.agent) agentNames.add(t.agent) }))
                      if (agentNames.size <= 1) return null
                      return (
                        <div className="flex flex-wrap gap-1">
                          <button
                            type="button"
                            onClick={() => setRunAgentFilter(null)}
                            className={cn('text-[10px] px-2 py-1 rounded border transition-colors', runAgentFilter === null ? 'bg-primary/15 text-primary border-primary/30' : 'border-border/40 text-muted-foreground hover:bg-muted/80')}
                          >
                            All agents
                          </button>
                          {Array.from(agentNames).sort().map(name => (
                            <button
                              key={name}
                              type="button"
                              onClick={() => setRunAgentFilter(runAgentFilter === name ? null : name)}
                              className={cn('text-[10px] px-2 py-1 rounded border transition-colors', runAgentFilter === name ? 'bg-primary/15 text-primary border-primary/30' : 'border-border/40 text-muted-foreground hover:bg-muted/80')}
                            >
                              {name}
                            </button>
                          ))}
                        </div>
                      )
                    })()}
                  </div>
                )}
                {runViewMode === 'list' ? (
                <>
                {runs
                  .filter(r => !runStatusFilter || r.status === runStatusFilter)
                  .filter(r => !runAgentFilter || r.tasks.some(t => t.agent === runAgentFilter))
                  .map(run => {
                const expanded = expandedRun === run.id
                const statusColor = statusDotColor(run.status)
                return (
                  <div key={run.id} className={cn('rounded-lg border border-border/40 transition-colors', expanded ? 'border-primary/40' : 'hover:bg-muted/20')}>
                    <button
                      type="button"
                      className="w-full flex items-center gap-2 px-3 py-2 text-left"
                      onClick={() => setExpandedRun(expanded ? null : run.id)}
                      aria-expanded={expanded}
                    >
                      <span className={cn('inline-block h-2 w-2 shrink-0 rounded-full', statusColor)} />
                      <span className="flex-1 truncate text-[11px]">{run.goal}</span>
                      <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">
                        {run.completed_count}/{run.completed_count + run.failed_count} tasks
                      </span>
                      {run.started_at && (
                        <span className="shrink-0 text-[10px] text-muted-foreground/70">
                          {formatElapsed(run.started_at, run.finished_at)}
                        </span>
                      )}
                      <span className="shrink-0 text-[10px] text-muted-foreground">
                        {run.started_at ? new Date(run.started_at).toLocaleString() : ''}
                      </span>
                    </button>
                    {expanded && (
                      <div className="border-t border-border/40 px-3 py-2 space-y-3">
                        {run.error && (
                          <StatusBanner variant="error" message={run.error} dismissible={false} />
                        )}
                        {run.tasks.length > 0 && (() => {
                          const agentStats = getAgentTaskStats(run.tasks)
                          const agentNames = Object.keys(agentStats)
                          return (
                            <div className="space-y-2">
                              {agentNames.length > 1 && (
                                <div className="flex flex-wrap gap-2">
                                  {agentNames.map(name => {
                                    const s = agentStats[name]
                                    return (
                                      <span key={name} className="text-[10px] px-2 py-1 rounded bg-muted text-muted-foreground font-medium">
                                        {name}: {s.completed}/{s.total}
                                        {s.failed > 0 && <span className="text-destructive ml-1">({s.failed} failed)</span>}
                                      </span>
                                    )
                                  })}
                                </div>
                              )}
                              <div className="space-y-1.5">
                                <p className="text-[10px] font-medium text-muted-foreground">Tasks ({run.tasks.length})</p>
                                {run.tasks.map(task => (
                                  <div key={task.id} className="flex items-center gap-2 rounded-lg border border-border/40 px-2.5 py-2 text-[11px]">
                                    <span className={cn('inline-block h-2 w-2 shrink-0 rounded-full', task.status === 'completed' ? 'bg-success' : task.status === 'failed' ? 'bg-destructive' : 'bg-muted-foreground/30')} />
                                    <span className="font-medium text-[10px] min-w-[64px] text-muted-foreground">{task.agent}</span>
                                    <span className="flex-1 truncate">{task.description}</span>
                                    {task.result_preview && (
                                      <span className="text-[10px] text-muted-foreground/50 truncate max-w-[200px]">{task.result_preview}</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )
                        })()}
                        {run.response && (
                          <div className="rounded-lg border border-border/40 bg-muted/30 p-3">
                            <p className="text-[10px] font-medium text-muted-foreground mb-1">Result</p>
                            <p className="text-[11px] whitespace-pre-wrap line-clamp-4">{run.response}</p>
                          </div>
                        )}
                        {run.logs.length > 0 && (
                          <div className="rounded-lg bg-muted/20 p-3">
                            <p className="text-[10px] font-medium text-muted-foreground mb-1">Logs ({run.logs.length})</p>
                            <pre className="text-[10px] text-muted-foreground whitespace-pre-wrap max-h-40 overflow-y-auto font-mono">
                              {run.logs.join('\n')}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
              </>
                ) : (
                  <div className="space-y-4">
                    {runs
                      .filter(r => !runStatusFilter || r.status === runStatusFilter)
                      .filter(r => !runAgentFilter || r.tasks.some(t => t.agent === runAgentFilter))
                      .map(run => {
                        const statusColor = statusTextColor(run.status)
                        return (
                          <div key={run.id} className="space-y-2">
                            <div className="flex items-center gap-2">
                              <span className={cn('text-[10px] font-medium rounded-full', statusColor)}>{run.status}</span>
                              <span className="text-[11px] flex-1 truncate">{run.goal}</span>
                              {run.started_at && (
                                <span className="text-[10px] text-muted-foreground/70">
                                  {formatElapsed(run.started_at, run.finished_at)}
                                </span>
                              )}
                              <span className="text-[10px] text-muted-foreground">{run.started_at ? new Date(run.started_at).toLocaleString() : ''}</span>
                            </div>
                            <div className="ml-2 border-l-2 border-border/40 pl-4 space-y-2">
                              {run.tasks.map((task, i) => {
                                const taskColor = taskDotColor(task.status)
                                return (
                                  <div key={task.id} className="relative flex items-center gap-1.5">
                                    <div className={cn('absolute -left-[21px] h-2.5 w-2.5 rounded-full border-2 border-background', taskColor)} />
                                    <span className="text-[10px] text-muted-foreground w-8 shrink-0">#{i + 1}</span>
                                    <span className="text-[10px] text-muted-foreground min-w-[64px] shrink-0">{task.agent}</span>
                                    <span className="text-[10px] flex-1 truncate">{task.description}</span>
                                    <span className={cn('text-[10px] px-1.5 py-0.5 rounded-full', taskBadgeStyle(task.status))}>
                                      {task.status}
                                    </span>
                                  </div>
                                )
                              })}
                              {run.tasks.length === 0 && (
                                <p className="text-[10px] text-muted-foreground/60 italic">No tasks recorded</p>
                              )}
                            </div>
                          </div>
                        )
                      })}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
        </SectionErrorBoundary>

      <AlertDialog open={pendingDelete !== null} onOpenChange={() => setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete agent?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &ldquo;{pendingDelete?.name}&rdquo;? This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => { if (pendingDelete) handleDelete(pendingDelete.id) }} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </PageContainer>
  )
}
