'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button, Input, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { agentsController, type Agent, type AgentRun } from '@/lib/agents-controller'
import { useToastStore } from '@/lib/toast-store'
import {
  ArrowLeft, Bot, Wrench, FileText, Play, History, Edit,
  Trash2, Loader2, CheckCircle, XCircle
} from 'lucide-react'

const TOOL_LABELS: Record<string, string> = {
  code_execution: 'Code Execution',
  file_read: 'File Read',
  file_search: 'File Search',
  web_search: 'Web Search',
  knowledge_retrieval: 'Knowledge Retrieval',
  image_analysis: 'Image Analysis',
  data_analysis: 'Data Analysis',
  citation: 'Citation',
}

export default function AgentDetailPage() {
  const params = useParams()
  const router = useRouter()
  const agentId = params.id as string
  const addToast = useToastStore(s => s.addToast)

  const [agent, setAgent] = useState<Agent | null>(null)
  const [runs, setRuns] = useState<AgentRun[]>([])
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')
  const [editInstructions, setEditInstructions] = useState('')
  const [editTools, setEditTools] = useState<string[]>([])
  const [saving, setSaving] = useState(false)

  // Execute state
  const [executing, setExecuting] = useState(false)
  const [execPrompt, setExecPrompt] = useState('')
  const [execResult, setExecResult] = useState<string | null>(null)

  const fetchAgent = useCallback(async () => {
    setLoading(true)
    try {
      const agents = await agentsController.list()
      const found = agents.find(a => a.id === agentId)
      if (found) {
        setAgent(found)
      } else {
        addToast('Agent not found', 'error')
        router.push('/agents')
      }

      const runsData = await agentsController.listRuns(10)
      setRuns(runsData.runs.filter(r => r.tasks.some(t => t.agent === agentId)))
    } catch {
      addToast('Failed to load agent', 'error')
    } finally {
      setLoading(false)
    }
  }, [agentId, addToast, router])

  useEffect(() => { fetchAgent() }, [fetchAgent])

  const handleSave = async () => {
    if (!agent) return
    setSaving(true)
    try {
      await agentsController.update(agent.id, {
        name: editName,
        description: editDesc,
        instructions: editInstructions,
        tools: editTools,
      })
      setAgent({ ...agent, name: editName, description: editDesc, instructions: editInstructions, tools: editTools })
      setEditing(false)
      addToast('Updated', 'success')
    } catch {
      addToast('Update failed', 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!agent) return
    try {
      await agentsController.delete(agent.id)
      addToast('Deleted', 'success')
      router.push('/agents')
    } catch {
      addToast('Delete failed', 'error')
    }
  }

  const handleExecute = async () => {
    if (!agent || !execPrompt.trim()) return
    setExecuting(true)
    setExecResult(null)
    try {
      const result = await agentsController.execute(agent.id, execPrompt)
      setExecResult(result.response)
      addToast('Execution complete', 'success')
    } catch {
      addToast('Execution failed', 'error')
    } finally {
      setExecuting(false)
    }
  }

  const toggleTool = (tool: string) => {
    setEditTools(prev =>
      prev.includes(tool) ? prev.filter(t => t !== tool) : [...prev, tool]
    )
  }

  return (
    <PageContainer
      title={agent ? agent.name : 'Loading...'}
      subtitle={agent?.description}
      loading={loading}
      loadingCards={3}
      headerRight={
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => router.push('/agents')}>
            <ArrowLeft className="h-4 w-4 mr-1" /> Back
          </Button>
          <Button variant="outline" size="sm" onClick={fetchAgent}>
            <IconRefresh className="h-4 w-4" />
          </Button>
        </div>
      }
    >
      {!loading && agent && (
        <div className="space-y-6">
          {/* Info Card */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Bot className="h-4 w-4" />
                  Agent Info
                </CardTitle>
                <div className="flex items-center gap-1">
                  {editing ? (
                    <>
                      <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setEditing(false)}>
                        Cancel
                      </Button>
                      <Button size="sm" className="h-6 text-[10px]" onClick={handleSave} disabled={saving}>
                        {saving ? '...' : 'Save'}
                      </Button>
                    </>
                  ) : (
                    <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => {
                      setEditing(true)
                      setEditName(agent.name)
                      setEditDesc(agent.description)
                      setEditInstructions(agent.instructions)
                      setEditTools([...agent.tools])
                    }}>
                      <Edit className="h-3 w-3 mr-1" /> Edit
                    </Button>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {editing ? (
                <div className="space-y-3">
                  <Input value={editName} onChange={e => setEditName(e.target.value)} placeholder="Name" className="h-8 text-sm" />
                  <Input value={editDesc} onChange={e => setEditDesc(e.target.value)} placeholder="Description" className="h-8 text-sm" />
                  <textarea
                    value={editInstructions}
                    onChange={e => setEditInstructions(e.target.value)}
                    placeholder="Instructions"
                    className="w-full h-24 p-2 text-sm border rounded-md resize-none bg-background"
                  />
                </div>
              ) : (
                <>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Name</span>
                    <span className="font-medium">{agent.name}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Description</span>
                    <span className="font-medium">{agent.description || '—'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">ID</span>
                    <span className="font-mono text-xs">{agent.id}</span>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          {/* Tools Card */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Wrench className="h-4 w-4" />
                Tools
              </CardTitle>
            </CardHeader>
            <CardContent>
              {editing ? (
                <div className="flex flex-wrap gap-2">
                  {Object.entries(TOOL_LABELS).map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => toggleTool(key)}
                      className={`px-2 py-1 text-xs rounded border transition-colors ${
                        editTools.includes(key)
                          ? 'bg-primary/15 text-primary border-primary/30'
                          : 'border-border text-muted-foreground hover:bg-muted/50'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {agent.tools.length === 0 ? (
                    <span className="text-xs text-muted-foreground">No tools assigned</span>
                  ) : (
                    agent.tools.map(tool => (
                      <Badge key={tool} variant="secondary">{TOOL_LABELS[tool] || tool}</Badge>
                    ))
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Instructions Card */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Instructions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{agent.instructions || 'No instructions set'}</p>
            </CardContent>
          </Card>

          {/* Execute Card */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Play className="h-4 w-4" />
                Execute
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input
                  value={execPrompt}
                  onChange={e => setExecPrompt(e.target.value)}
                  placeholder="Enter a prompt to execute..."
                  className="flex-1 h-8 text-sm"
                  onKeyDown={e => { if (e.key === 'Enter') handleExecute() }}
                />
                <Button size="sm" className="h-8 text-xs" onClick={handleExecute} disabled={executing || !execPrompt.trim()}>
                  {executing ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Run'}
                </Button>
              </div>
              {execResult && (
                <div className="p-3 bg-muted/50 rounded-md text-sm whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {execResult}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Recent Runs */}
          {runs.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <History className="h-4 w-4" />
                  Recent Runs
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {runs.slice(0, 5).map(run => (
                  <div key={run.id} className="flex items-center gap-2 text-xs">
                    {run.status === 'completed' && <CheckCircle className="h-3 w-3 text-green-500" />}
                    {run.status === 'failed' && <XCircle className="h-3 w-3 text-red-500" />}
                    {run.status === 'running' && <Loader2 className="h-3 w-3 text-blue-500 animate-spin" />}
                    <span className="flex-1 truncate">{run.goal}</span>
                    <span className="text-muted-foreground">{run.completed_count}/{run.tasks.length} tasks</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Actions */}
          <Card>
            <CardContent className="py-3">
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="h-7 text-[11px]" onClick={() => router.push(`/chat?agent=${agent.id}`)}>
                  <Play className="h-3 w-3 mr-1" /> Chat with Agent
                </Button>
                <Button variant="outline" size="sm" className="h-7 text-[11px] text-destructive border-destructive/30" onClick={handleDelete}>
                  <Trash2 className="h-3 w-3 mr-1" /> Delete
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </PageContainer>
  )
}
