'use client'
import { notify } from '@/lib/notify'

/**
 * Stateful core of the Agents page.
 *
 * Owns IO and state only — the agents/runs fetches, the create/edit/delete
 * mutations with their optimistic update and undo toast, bulk operations,
 * agent execution and the multi-agent orchestration with its SSE phase
 * stream. The agent-* panels own render.
 *
 * Page-local rather than in `hooks/` for the same reason as the VM hook:
 * it depends on the page-local constants/format siblings and has no
 * `useAgents*` family up in `hooks/` to join.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter } from '@/vite/next-compat/navigation'
import {
  agentsController,
  type Agent,
  type OrchestrateTask,
  type AgentRun,
} from '@/lib/agents-controller'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
import { downloadJson } from '@/lib/download-utils'
import { todayDateString } from '@/lib/format-bytes'
import { agentSchema, agentExecuteSchema, orchestrateSchema } from '@/lib/validation-schemas'
import { logger } from '@/lib/dev-log'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

export function useAgentsWorkbench() {
  const router = useRouter()
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

  const [createErrors, setCreateErrors] = useState<{
    name?: string
    description?: string
    instructions?: string
  }>({})
  const [editErrors, setEditErrors] = useState<{
    name?: string
    description?: string
    instructions?: string
  }>({})
  const [execErrors, setExecErrors] = useState<{ prompt?: string }>({})
  const [orchErrors, setOrchErrors] = useState<{ goal?: string; context?: string }>({})

  const fetchAgents = useCallback(async () => {
    setLoading(true)
    try {
      const data = await agentsController.list()
      setAgents(data)
    } catch {
      notify.toast('Could not load agents', { type: 'error' })
    }
    setLoading(false)
  }, [])

  useRefreshShortcut(fetchAgents)

  useEffect(() => {
    fetchAgents()
  }, [fetchAgents])

  const fetchRuns = useCallback(async () => {
    setRunsLoading(true)
    try {
      const res = await agentsController.listRuns(20)
      setRuns(res.runs || [])
    } catch {
      notify.toast('Could not load run history', { type: 'error' })
    }
    setRunsLoading(false)
  }, [])

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      )
        return
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
    if (agents.length === 0) return notify.toast('No agents to export', { type: 'error' })
    const data = agents.map((a) => ({
      name: a.name,
      description: a.description,
      instructions: a.instructions,
      tools: a.tools,
    }))
    downloadJson(data, `agents-${todayDateString()}.json`)
    notify.toast(`Exported ${agents.length} agents`, { type: 'success' })
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
          } catch {
            /* skip individual failures */
          }
        }
        notify.toast(`Imported ${count} agents`, { type: 'success' })
        await fetchAgents()
      } catch {
        notify.toast('Invalid agent file', { type: 'error' })
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  const toggleTool = (tool: string, current: string[], setter: (v: string[]) => void) => {
    setter(current.includes(tool) ? current.filter((t) => t !== tool) : [...current, tool])
  }

  const handleCreate = async () => {
    const result = agentSchema.safeParse({
      name: newName,
      description: newDesc,
      instructions: newInstructions,
    })
    if (!result.success) {
      const fieldErrors: { name?: string; description?: string; instructions?: string } = {}
      result.error.issues.forEach((issue) => {
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
    setAgents((prev) => [optimistic, ...prev])
    setNewName('')
    setNewDesc('')
    setNewInstructions('')
    setNewTools([])
    try {
      const created = await agentsController.create({
        name: trimmedName,
        description: trimmedDesc,
        instructions: trimmedInstructions,
        tools: newTools,
      })
      setAgents((prev) => prev.map((a) => (a.id === tempId ? created : a)))
      notify.toast('Agent created', { type: 'success' })
    } catch {
      setAgents((prev) => prev.filter((a) => a.id !== tempId))
      notify.toast('Could not create agent', { type: 'error' })
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
    setEditName('')
    setEditDesc('')
    setEditInstructions('')
    setEditTools([])
  }

  const handleUpdate = async (id: string) => {
    const result = agentSchema.safeParse({
      name: editName,
      description: editDesc,
      instructions: editInstructions,
    })
    if (!result.success) {
      const fieldErrors: { name?: string; description?: string; instructions?: string } = {}
      result.error.issues.forEach((issue) => {
        const field = issue.path[0] as string
        if (field === 'name') fieldErrors.name = issue.message
        if (field === 'description') fieldErrors.description = issue.message
        if (field === 'instructions') fieldErrors.instructions = issue.message
      })
      setEditErrors(fieldErrors)
      return
    }
    setEditErrors({})
    const oldAgent = agents.find((a) => a.id === id)
    const trimmedName = editName.trim()
    const trimmedDesc = editDesc.trim()
    const trimmedInstructions = editInstructions.trim()
    setAgents((prev) =>
      prev.map((a) =>
        a.id === id
          ? {
              ...a,
              name: trimmedName,
              description: trimmedDesc,
              instructions: trimmedInstructions,
              tools: editTools,
            }
          : a,
      ),
    )
    setEditingId(null)
    try {
      await agentsController.update(id, {
        name: trimmedName,
        description: trimmedDesc,
        instructions: trimmedInstructions,
        tools: editTools,
      })
      notify.toast('Agent updated', { type: 'success' })
    } catch {
      if (oldAgent) setAgents((prev) => prev.map((a) => (a.id === id ? oldAgent : a)))
      notify.toast('Could not update', { type: 'error' })
    }
  }

  const handleDelete = async (id: string) => {
    const deleted = agents.find((a) => a.id === id)
    if (!deleted) return
    setAgents((prev) => prev.filter((a) => a.id !== id))
    try {
      await agentsController.delete(id)
      notify.toast('Agent deleted', {
        type: 'success',
        onUndo: () => {
          setAgents((prev) => [deleted, ...prev])
          agentsController
            .create({
              name: deleted.name,
              description: deleted.description,
              instructions: deleted.instructions,
              tools: deleted.tools,
            })
            .catch((e) => {
              logger.warning('Could not restore agent', { exception: String(e) })
              addToast('Could not restore agent', 'error')
              fetchAgents()
            })
        },
      })
    } catch {
      setAgents((prev) => [deleted, ...prev])
      notify.toast('Could not delete agent', { type: 'error' })
    }
  }

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
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
      setSelectedIds(new Set(filteredAgents.map((a) => a.id)))
    }
  }

  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) return
    const deletedIds = new Set(selectedIds)
    const deletedAgents = agents.filter((a) => deletedIds.has(a.id))
    setAgents((prev) => prev.filter((a) => !deletedIds.has(a.id)))
    setSelectedIds(new Set())
    try {
      await Promise.all(Array.from(deletedIds).map((id) => agentsController.delete(id)))
      notify.toast(`Deleted ${deletedIds.size} agents`, { type: 'success' })
    } catch {
      setAgents((prev) => [...deletedAgents, ...prev])
      notify.toast('Could not delete some agents', { type: 'error' })
    }
  }

  const handleBulkExport = () => {
    const selected = agents.filter((a) => selectedIds.has(a.id))
    downloadJson(selected, `agents-export-${todayDateString()}.json`)
    notify.toast(`Exported ${selected.length} agents`, { type: 'success' })
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
      result.error.issues.forEach((issue) => {
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

    await agentsController.orchestrate(
      orchGoal,
      orchContext,
      {
        onPlan: (tasks) => {
          setOrchTasks(tasks)
          const statuses: Record<string, string> = {}
          tasks.forEach((t) => {
            statuses[t.id] = 'pending'
          })
          setOrchTaskStatuses(statuses)
          setOrchPhase('EXECUTE')
        },
        onLevelChange: (level, total) => {
          setOrchLevel(level + 1)
          setOrchTotalLevels(total)
        },
        onTaskStatus: (taskId, status) => {
          setOrchTaskStatuses((prev) => ({ ...prev, [taskId]: status }))
        },
        onCompose: () => {
          setOrchPhase('COMPOSE')
        },
        onComplete: (response) => {
          setOrchResponse(response)
          setOrchPhase('COMPLETE')
          setOrchRunning(false)
          notify.toast('Orchestration complete', { type: 'success' })
          fetchRuns()
        },
        onError: (error) => {
          setOrchError(error)
          setOrchPhase('ERROR')
          setOrchRunning(false)
          notify.toast(error, { type: 'error' })
          fetchRuns()
        },
      },
      undefined,
      orchAgentIds.length > 0 ? orchAgentIds : undefined,
    )
  }

  const toolCount = agents.reduce((acc, a) => acc + (a.tools?.length ?? 0), 0)

  const filteredAgents = agentSearch
    ? agents.filter(
        (a) =>
          a.name.toLowerCase().includes(agentSearch.toLowerCase()) ||
          (a.description?.toLowerCase() ?? '').includes(agentSearch.toLowerCase()),
      )
    : agents

  const handleClone = async (agent: Agent) => {
    try {
      await agentsController.create({
        name: `${agent.name} (copy)`,
        description: agent.description,
        instructions: agent.instructions,
        tools: [...agent.tools],
      })
      notify.toast(`Cloned "${agent.name}"`, { type: 'success' })
      await fetchAgents()
    } catch {
      notify.toast('Could not clone agent', { type: 'error' })
    }
  }

  return {
    router,
    agents,
    loading,
    newName,
    setNewName,
    newDesc,
    setNewDesc,
    newInstructions,
    setNewInstructions,
    newTools,
    setNewTools,
    creating,
    editingId,
    setEditingId,
    editName,
    setEditName,
    editDesc,
    setEditDesc,
    editInstructions,
    setEditInstructions,
    editTools,
    setEditTools,
    execAgentId,
    setExecAgentId,
    execPrompt,
    setExecPrompt,
    execResult,
    setExecResult,
    execToolsUsed,
    setExecToolsUsed,
    pendingDelete,
    setPendingDelete,
    selectedIds,
    setSelectedIds,
    execRunning,
    setExecRunning,
    agentSearch,
    setAgentSearch,
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
    runs,
    runsLoading,
    expandedRun,
    setExpandedRun,
    runStatusFilter,
    setRunStatusFilter,
    runAgentFilter,
    setRunAgentFilter,
    runViewMode,
    setRunViewMode,
    createErrors,
    setCreateErrors,
    setEditErrors,
    setExecErrors,
    setOrchErrors,
    setOrchResponse,
    setOrchError,
    setOrchPhase,
    setOrchTasks,
    setOrchTaskStatuses,
    editErrors,
    execErrors,
    orchErrors,
    fetchAgents,
    fetchRuns,
    handleExport,
    handleImport,
    toggleTool,
    handleCreate,
    startEdit,
    cancelEdit,
    handleUpdate,
    handleExecute,
    handleDelete,
    toggleSelect,
    selectAll,
    handleBulkDelete,
    handleBulkExport,
    handleClone,
    handleOrchestrate,
    toolCount,
    filteredAgents,
    importInputRef,
  }
}

/** The contract the Agents panels render against. */
export type AgentsWorkbench = ReturnType<typeof useAgentsWorkbench>
