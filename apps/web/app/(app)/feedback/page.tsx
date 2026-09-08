'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, cn, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { feedbackController, type FeedbackStats, type WorkflowStatus, type TrainingStats } from '@/lib/feedback-controller'
import { FeedbackInsightsCard } from '@/components/feedback/FeedbackInsightsCard'
import { WorkflowSection } from '@/components/workflow/WorkflowSection'
import { feedbackConversationsController } from '@/lib/feedback-conversations-controller'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

type Tab = 'stats' | 'conversations' | 'training'

export default function FeedbackPage() {
  const router = useRouter()
  const [tab, setTab] = useState<Tab>('stats')
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const [stats, setStats] = useState<FeedbackStats | null>(null)
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null)
  const [trainStats, setTrainStats] = useState<TrainingStats | null>(null)
  const [conversations, setConversations] = useState<Awaited<ReturnType<typeof feedbackConversationsController.list>>>([])
  const [newConvName, setNewConvName] = useState('')
  const [creating, setCreating] = useState(false)
  const [workflowBusy, setWorkflowBusy] = useState(false)
  const [convSearch, setConvSearch] = useState('')

  useEffect(() => {
    Promise.all([
      feedbackController.getFeedbackStats().catch((e) => { logger.warning('Could not feedback stats', e); return null }),
      feedbackController.getWorkflowStatus().catch((e) => { logger.warning('Could not workflow status', e); return null }),
      feedbackController.getTrainingStats().catch((e) => { logger.warning('Could not load training stats', e); return null }),
    ]).then(([s, w, t]) => {
      setStats(s)
      setWorkflow(w)
      setTrainStats(t)
      if (!s && !w && !t) setLoadError('Could not load feedback data. Please try again.')
    }).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void handleRefreshStats() }
      if (e.key === 'n' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); setTab('conversations') }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [])

  const handleExport = useCallback(async () => {
    try {
      const resp = await feedbackController.exportTrainingData('json')
      addToast(`Exported ${resp.count} items to ${resp.path ?? 'server'}`, 'success')
    } catch {
      addToast('Could not export training data', 'error')
    }
  }, [addToast])

  const handleRefreshStats = async () => {
    const [s, w, t] = await Promise.all([
      feedbackController.getFeedbackStats().catch((e) => { logger.warning('Could not feedback stats refresh', e); return null }),
      feedbackController.getWorkflowStatus().catch((e) => { logger.warning('Could not workflow status refresh', e); return null }),
      feedbackController.getTrainingStats().catch((e) => { logger.warning('Could not refresh training stats', e); return null }),
    ])
    setStats(s)
    setWorkflow(w)
    setTrainStats(t)
  }

  const handleLoadConversations = async () => {
    try {
      setConversations(await feedbackConversationsController.list())
    } catch {
      addToast('Could not load conversations', 'error')
    }
  }

  const handleCreateConversation = async () => {
    if (!newConvName.trim()) return
    setCreating(true)
    try {
      await feedbackConversationsController.create(newConvName)
      setNewConvName('')
      await handleLoadConversations()
    } catch {
      addToast('Could not create conversation', 'error')
    } finally {
      setCreating(false)
    }
  }

  const handleDeleteConversation = async (id: string) => {
    try {
      await feedbackConversationsController.delete(id)
      await handleLoadConversations()
    } catch {
      addToast('Could not delete conversation', 'error')
    }
  }

  const handleTogglePin = async (conv: { id: string; pinned: boolean }) => {
    try {
      await feedbackConversationsController.togglePin(conv.id, !conv.pinned)
      await handleLoadConversations()
    } catch {
      addToast('Could not update pin', 'error')
    }
  }

  const handleToggleStar = async (conv: { id: string; starred: boolean }) => {
    try {
      await feedbackConversationsController.toggleStar(conv.id, !conv.starred)
      await handleLoadConversations()
    } catch {
      addToast('Could not update star', 'error')
    }
  }

  if (loading) {
    return (
      <PageContainer title="Feedback" subtitle="Analytics & management">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}><CardContent className="p-4 space-y-2"><Skeleton className="h-3 w-16" /><Skeleton className="h-5 w-12" /></CardContent></Card>
          ))}
        </div>
        <Card><CardContent className="p-4 space-y-3"><Skeleton className="h-4 w-32" /><Skeleton className="h-24 w-full rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Feedback" subtitle="Analytics & management" error={loadError} onRetry={handleRefreshStats}>
      <div className="flex gap-0.5 border-b border-border/20 pb-0">
        {(['stats', 'conversations', 'training'] as Tab[]).map(t => (
          <button
            type="button"
            role="tab"
            key={t}
            aria-selected={tab === t}
            aria-label={`${t.charAt(0).toUpperCase() + t.slice(1)} tab`}
            onClick={() => {
              setTab(t)
              if (t === 'conversations') handleLoadConversations()
            }}
            className={cn('px-2.5 py-1 text-[10px] font-medium rounded-t transition-colors', tab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground/60 hover:text-foreground')}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'stats' && (
        <>
          <KpiGrid>
            <StatCard label="Thumbs Up" value={String(stats?.db_stats?.thumbs_up ?? 0)} />
            <StatCard label="Thumbs Down" value={String(stats?.db_stats?.thumbs_down ?? 0)} />
            <StatCard label="Total Feedback" value={String(stats?.db_stats?.feedback_total ?? 0)} />
            <StatCard label="Up Ratio" value={`${((stats?.db_stats?.ratio ?? 0) * 100).toFixed(1)}%`} />
          </KpiGrid>

          <FeedbackInsightsCard stats={stats} />

          {workflow && (
            <KpiGrid>
              <StatCard label="Workflow Status" value={workflow.running ? 'Running' : 'Stopped'} />
              <StatCard label="Workflow Runs" value={String(workflow.stats?.workflow_runs ?? 0)} />
              <StatCard label="Aggregations" value={String(workflow.stats?.aggregations_performed ?? 0)} />
              <StatCard label="Prunes" value={String(workflow.stats?.prunes_performed ?? 0)} />
            </KpiGrid>
          )}
        </>
      )}

      {tab === 'conversations' && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Conversations ({conversations.length})</CardTitle>
            <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={handleLoadConversations} aria-label="Refresh conversations">
              <IconRefresh className="h-3 w-3" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-2 px-2.5 pb-2.5">
            <div className="flex gap-1.5">
              <Input
                value={newConvName}
                onChange={e => setNewConvName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCreateConversation()}
                placeholder="New conversation name..."
                className="h-7 text-[11px] flex-1"
              />
              <Button size="sm" className="h-7 text-[11px]" onClick={handleCreateConversation} disabled={creating || !newConvName.trim()}>
                Create
              </Button>
            </div>
            {conversations.length > 0 && (
              <Input
                value={convSearch}
                onChange={e => setConvSearch(e.target.value)}
                placeholder="Search conversations..."
                aria-label="Search conversations"
                className="h-7 text-[11px]"
              />
            )}
            {conversations.length === 0 ? (
              <p className="text-[10px] text-muted-foreground/60">No conversations yet.</p>
            ) : (
              <div className="space-y-0.5 max-h-96 overflow-y-auto">
                {conversations
                  .filter(c => !convSearch || c.name.toLowerCase().includes(convSearch.toLowerCase()))
                  .map(conv => (
                  <div key={conv.id} className="flex items-center justify-between rounded-lg border border-border/40 px-2.5 py-2 text-[11px] group hover:bg-muted/20 transition-colors">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="font-medium truncate">{conv.name}</span>
                        {conv.pinned && <span className="text-[9px] bg-primary/10 text-primary px-1.5 py-0.5 rounded-full">pinned</span>}
                        {conv.starred && <span className="text-[9px] bg-warning/10 text-warning px-1.5 py-0.5 rounded-full">starred</span>}
                      </div>
                      <div className="text-[10px] text-muted-foreground/60 mt-0.5 font-mono tabular-nums">
                        {conv.message_count} messages · {conv.created_at ? new Date(conv.created_at).toLocaleDateString() : '—'}
                      </div>
                    </div>
                    <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                      <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => handleTogglePin(conv)}>
                        {conv.pinned ? 'Unpin' : 'Pin'}
                      </Button>
                      <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => handleToggleStar(conv)}>
                        {conv.starred ? 'Unstar' : 'Star'}
                      </Button>
                      <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={() => handleDeleteConversation(conv.id)}>
                        Delete
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'training' && (
        <>
          <Card>
            <CardHeader className="pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-[11px] font-medium">Training Data</CardTitle>
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5">
              {trainStats ? (
                <KpiGrid>
                  <StatCard label="Training Jobs" value={String(trainStats.feedback_pairs ?? 0)} />
                  <StatCard label="Last Training" value={trainStats.last_training ? new Date(trainStats.last_training).toLocaleDateString() : 'Never'} />
                  <StatCard label="Final Loss" value={trainStats.quality_score != null ? trainStats.quality_score.toFixed(3) : '—'} />
                </KpiGrid>
              ) : (
                <div className="text-center py-3 text-[10px] text-muted-foreground/60">
                  No training data available.
                  <div className="mt-1.5">
                    <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={() => router.push('/training')}>
                      Go to Training
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {workflow && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
                <CardTitle className="text-[11px] font-medium">Workflow Controls</CardTitle>
                <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={handleRefreshStats} aria-label="Refresh stats">
                  <IconRefresh className="h-3 w-3" />
                </Button>
              </CardHeader>
              <CardContent className="space-y-2 px-2.5 pb-2.5">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                  {[
                    { label: 'Status', value: workflow.running ? 'Running' : 'Stopped', color: workflow.running ? 'text-success' : 'text-muted-foreground/60' },
                    { label: 'Workflow Runs', value: String(workflow.stats?.workflow_runs ?? 0) },
                    { label: 'Aggregations', value: String(workflow.stats?.aggregations_performed ?? 0) },
                    { label: 'Prunes', value: String(workflow.stats?.prunes_performed ?? 0) },
                  ].map(s => (
                    <div key={s.label} className="rounded-lg bg-muted/20 p-2 text-center">
                      <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">{s.label}</div>
                      <div className={cn('text-[11px] font-mono font-medium tabular-nums mt-0.5', s.color ?? '')}>{s.value}</div>
                    </div>
                  ))}
                </div>
                {workflow.stats?.feedback_recorded != null && (
                  <div className="flex flex-wrap gap-2 text-[10px] text-muted-foreground/60">
                    <span>Feedback recorded: {workflow.stats.feedback_recorded}</span>
                    <span>Exports: {workflow.stats.exports_performed ?? 0}</span>
                  </div>
                )}
                <div className="flex gap-1">
                  <Button size="sm" className="h-7 text-[11px]" disabled={workflowBusy} onClick={async () => { setWorkflowBusy(true); try { await feedbackController.triggerWorkflowAction('aggregate'); addToast('Aggregation triggered', 'success'); handleRefreshStats() } catch { addToast('aggregation', 'error') } finally { setWorkflowBusy(false) } }}>
                    Aggregate
                  </Button>
                  <Button size="sm" variant="outline" className="h-7 text-[11px]" disabled={workflowBusy} onClick={async () => { setWorkflowBusy(true); try { await feedbackController.triggerWorkflowAction('prune'); addToast('Prune triggered', 'success'); handleRefreshStats() } catch { addToast('prune', 'error') } finally { setWorkflowBusy(false) } }}>
                    Prune
                  </Button>
                  <Button size="sm" variant="outline" className="h-7 text-[11px]" disabled={workflowBusy} onClick={async () => { setWorkflowBusy(true); try { await feedbackController.triggerWorkflowAction('export'); addToast('Export triggered', 'success') } catch { addToast('export', 'error') } finally { setWorkflowBusy(false) } }}>
                    Export
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          <WorkflowSection />
        </>
      )}
    </PageContainer>
  )
}
