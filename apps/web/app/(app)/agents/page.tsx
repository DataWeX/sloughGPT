'use client'

import { useState, useEffect } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from '@/components/ui/dialog'
import {
  AlertDialog, AlertDialogContent, AlertDialogHeader, AlertDialogTitle,
  AlertDialogDescription, AlertDialogAction, AlertDialogCancel,
} from '@/components/ui/alert-dialog'
import { agentsController } from '@/lib/agents-controller'
import { cn } from '@/lib/cn'

interface Agent {
  id: string
  name: string
  description: string
  instructions: string
  tools: string[]
  avatar: string
}

const ALL_TOOLS = [
  { id: 'code_execution', label: 'Code Execution' },
  { id: 'file_search', label: 'File Search' },
  { id: 'web_search', label: 'Web Search' },
  { id: 'citation', label: 'Citation' },
  { id: 'memory', label: 'Memory' },
]

const emptyForm = { name: '', description: '', instructions: '', tools: [] as string[], avatar: '' }

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [editAgent, setEditAgent] = useState<Agent | null>(null)
  const [form, setForm] = useState(emptyForm)
  const [saving, setSaving] = useState(false)
  const [deleteId, setDeleteId] = useState<string | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [runInput, setRunInput] = useState('')
  const [runResult, setRunResult] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  const fetchAgents = async () => {
    try {
      setLoading(true)
      const data = await agentsController.list()
      setAgents(data)
    } catch { /* ignore */ } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchAgents() }, [])

  const openCreate = () => {
    setEditAgent(null)
    setForm(emptyForm)
  }

  const openEdit = (agent: Agent) => {
    setEditAgent(agent)
    setForm({
      name: agent.name,
      description: agent.description,
      instructions: agent.instructions,
      tools: [...agent.tools],
      avatar: agent.avatar,
    })
  }

  const save = async () => {
    if (!form.name.trim()) return
    setSaving(true)
    try {
      if (editAgent) {
        const updated = await agentsController.update(editAgent.id, form)
        setAgents(prev => prev.map(a => a.id === editAgent.id ? { ...a, ...updated } : a))
      } else {
        const created = await agentsController.create(form)
        setAgents(prev => [...prev, created])
      }
      setEditAgent(null)
    } catch { /* ignore */ } finally {
      setSaving(false)
    }
  }

  const confirmDelete = async () => {
    if (!deleteId) return
    try {
      await agentsController.delete(deleteId)
      setAgents(prev => prev.filter(a => a.id !== deleteId))
    } catch { /* ignore */ }
    setDeleteId(null)
  }

  const runAgent = async () => {
    if (!runId || !runInput.trim()) return
    setRunning(true)
    setRunResult(null)
    try {
      const res = await agentsController.execute(runId, runInput)
      setRunResult(res.response)
    } catch { /* ignore */ } finally {
      setRunning(false)
    }
  }

  const toggleTool = (toolId: string) => {
    setForm(prev => ({
      ...prev,
      tools: prev.tools.includes(toolId)
        ? prev.tools.filter(t => t !== toolId)
        : [...prev.tools, toolId],
    }))
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Agent Builder" />}
        right={
          <Dialog onOpenChange={open => { if (open) openCreate(); if (!open) setEditAgent(null) }}>
            <DialogTrigger asChild>
              <Button size="sm">New Agent</Button>
            </DialogTrigger>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle className="text-base">{editAgent ? 'Edit Agent' : 'New Agent'}</DialogTitle>
              </DialogHeader>
              <AgentForm form={form} setForm={setForm} saving={saving} onSave={save} onCancel={() => setEditAgent(null)} />
            </DialogContent>
          </Dialog>
        }
      />

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Agents</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map(i => (
                  <div key={i} className="h-20 animate-pulse rounded-lg bg-muted" />
                ))}
              </div>
            ) : agents.length === 0 ? (
              <p className="text-sm text-muted-foreground py-8 text-center">
                No agents yet. Create one to get started.
              </p>
            ) : (
              <div className="space-y-3">
                {agents.map(agent => (
                  <div key={agent.id} className="rounded-lg border border-border p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-medium">{agent.name}</span>
                          <span className="text-xs text-muted-foreground font-mono">{agent.id}</span>
                        </div>
                        {agent.description && (
                          <p className="text-xs text-muted-foreground mb-2">{agent.description}</p>
                        )}
                        {agent.tools.length > 0 && (
                          <div className="flex flex-wrap gap-1.5">
                            {agent.tools.map(tool => (
                              <Badge key={tool} variant="secondary" className="text-[10px]">
                                {ALL_TOOLS.find(t => t.id === tool)?.label ?? tool}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button size="sm" variant="ghost" onClick={() => openEdit(agent)}>Edit</Button>
                          </DialogTrigger>
                          <DialogContent className="max-w-lg">
                            <DialogHeader>
                              <DialogTitle className="text-base">Edit Agent</DialogTitle>
                            </DialogHeader>
                            <AgentForm form={form} setForm={setForm} saving={saving} onSave={save} onCancel={() => setEditAgent(null)} />
                          </DialogContent>
                        </Dialog>
                        <Button size="sm" variant="ghost" onClick={() => { setRunId(agent.id); setRunResult(null); setRunInput('') }}>
                          Run
                        </Button>
                        <AlertDialog open={deleteId === agent.id} onOpenChange={open => !open && setDeleteId(null)}>
                          <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" onClick={() => setDeleteId(agent.id)}>
                            Delete
                          </Button>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>Delete Agent</AlertDialogTitle>
                              <AlertDialogDescription>
                                Are you sure you want to delete <span className="font-medium">{agent.name}</span>? This cannot be undone.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <div className="flex justify-end gap-2">
                              <AlertDialogCancel onClick={() => setDeleteId(null)}>Cancel</AlertDialogCancel>
                              <AlertDialogAction onClick={confirmDelete} className="bg-destructive text-destructive-foreground hover:opacity-90">
                                Delete
                              </AlertDialogAction>
                            </div>
                          </AlertDialogContent>
                        </AlertDialog>
                      </div>
                    </div>
                    {agent.instructions && (
                      <details className="mt-2">
                        <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                          Instructions
                        </summary>
                        <pre className="mt-1 text-xs text-muted-foreground whitespace-pre-wrap font-mono bg-muted/50 rounded p-2">
                          {agent.instructions}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {runId && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Run: {agents.find(a => a.id === runId)?.name ?? runId}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={runInput}
                onChange={e => setRunInput(e.target.value)}
                placeholder="Enter your request..."
                rows={3}
              />
              <div className="flex items-center gap-2">
                <Button size="sm" onClick={runAgent} disabled={running || !runInput.trim()}>
                  {running ? 'Running...' : 'Execute'}
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setRunId(null)}>Close</Button>
              </div>
              {runResult && (
                <div className="rounded-lg border border-border bg-muted/30 p-3">
                  <p className="text-xs font-medium text-muted-foreground mb-1">Response</p>
                  <p className="text-sm whitespace-pre-wrap">{runResult}</p>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

function AgentForm({
  form, setForm, saving, onSave, onCancel,
}: {
  form: typeof emptyForm
  setForm: (f: typeof emptyForm) => void
  saving: boolean
  onSave: () => void
  onCancel: () => void
}) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>Name</Label>
        <Input
          value={form.name}
          onChange={e => setForm({ ...form, name: e.target.value })}
          placeholder="My Agent"
        />
      </div>
      <div className="space-y-2">
        <Label>Description</Label>
        <Input
          value={form.description}
          onChange={e => setForm({ ...form, description: e.target.value })}
          placeholder="What this agent does"
        />
      </div>
      <div className="space-y-2">
        <Label>Instructions</Label>
        <Textarea
          value={form.instructions}
          onChange={e => setForm({ ...form, instructions: e.target.value })}
          placeholder="System prompt / instructions for the agent"
          rows={4}
        />
      </div>
      <div className="space-y-2">
        <Label>Tools</Label>
        <div className="flex flex-wrap gap-2">
          {ALL_TOOLS.map(tool => (
            <button
              key={tool.id}
              type="button"
              onClick={() => setForm({
                ...form,
                tools: form.tools.includes(tool.id)
                  ? form.tools.filter(t => t !== tool.id)
                  : [...form.tools, tool.id],
              })}
              className={cn(
                'px-3 py-1.5 text-xs rounded-md border transition-colors',
                form.tools.includes(tool.id)
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:border-primary/50',
              )}
            >
              {tool.label}
            </button>
          ))}
        </div>
      </div>
      <div className="flex justify-end gap-2 pt-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>Cancel</Button>
        <Button size="sm" onClick={onSave} disabled={saving || !form.name.trim()}>
          {saving ? 'Saving...' : 'Save'}
        </Button>
      </div>
    </div>
  )
}
