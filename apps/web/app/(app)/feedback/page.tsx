'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { feedbackController, type FeedbackStats, type WorkflowStatus, type TrainingStats } from '@/lib/feedback-controller'
import { FeedbackInsightsCard } from '@/components/feedback/FeedbackInsightsCard'
import { WorkflowSection } from '@/components/workflow/WorkflowSection'
import { feedbackConversationsController } from '@/lib/feedback-conversations-controller'
import { useToastStore } from '@/lib/toast-store'

type Tab = 'stats' | 'conversations' | 'training'

export default function FeedbackPage() {
  const [tab, setTab] = useState<Tab>('stats')
  const [loading, setLoading] = useState(true)
  const addToast = useToastStore(s => s.addToast)

  const [stats, setStats] = useState<FeedbackStats | null>(null)
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null)
  const [trainStats, setTrainStats] = useState<TrainingStats | null>(null)
  const [conversations, setConversations] = useState<Awaited<ReturnType<typeof feedbackConversationsController.list>>>([])
  const [newConvName, setNewConvName] = useState('')
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    Promise.all([
      feedbackController.getFeedbackStats().catch(() => null),
      feedbackController.getWorkflowStatus().catch(() => null),
      feedbackController.getTrainingStats().catch(() => null),
    ]).then(([s, w, t]) => {
      setStats(s)
      setWorkflow(w)
      setTrainStats(t)
    }).finally(() => setLoading(false))
  }, [])

  const handleRefreshStats = async () => {
    const [s, w, t] = await Promise.all([
      feedbackController.getFeedbackStats().catch(() => null),
      feedbackController.getWorkflowStatus().catch(() => null),
      feedbackController.getTrainingStats().catch(() => null),
    ])
    setStats(s)
    setWorkflow(w)
    setTrainStats(t)
  }

  const handleLoadConversations = async () => {
    try {
      setConversations(await feedbackConversationsController.list())
    } catch {
      addToast('Failed to load conversations', 'error')
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
      addToast('Failed to create conversation', 'error')
    } finally {
      setCreating(false)
    }
  }

  const handleDeleteConversation = async (id: string) => {
    try {
      await feedbackConversationsController.delete(id)
      await handleLoadConversations()
    } catch {
      addToast('Failed to delete conversation', 'error')
    }
  }

  const handleTogglePin = async (conv: { id: string; pinned: boolean }) => {
    try {
      await feedbackConversationsController.togglePin(conv.id, !conv.pinned)
      await handleLoadConversations()
    } catch {
      addToast('Failed to update pin', 'error')
    }
  }

  const handleToggleStar = async (conv: { id: string; starred: boolean }) => {
    try {
      await feedbackConversationsController.toggleStar(conv.id, !conv.starred)
      await handleLoadConversations()
    } catch {
      addToast('Failed to update star', 'error')
    }
  }

  if (loading) {
    return (
      <PageContainer title="Feedback" subtitle="Analytics & management" loadingCards={1}>
        <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Feedback" subtitle="Analytics & management">
      <div className="flex gap-1 border-b border-border/30 pb-0">
        {(['stats', 'conversations', 'training'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => {
              setTab(t)
              if (t === 'conversations') handleLoadConversations()
            }}
            className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors ${
              tab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
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

          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Feedback Summary</CardTitle>
              <Button size="sm" variant="ghost" onClick={handleRefreshStats}>
                <IconRefresh className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              {stats ? (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    { label: 'Thumbs Up', value: String(stats.db_stats?.thumbs_up ?? 0), color: 'text-success' },
                    { label: 'Thumbs Down', value: String(stats.db_stats?.thumbs_down ?? 0), color: 'text-destructive' },
                    { label: 'Total', value: String(stats.db_stats?.feedback_total ?? 0), color: '' },
                    { label: 'Up Ratio', value: `${((stats.db_stats?.ratio ?? 0) * 100).toFixed(1)}%`, color: 'text-primary' },
                  ].map(s => (
                    <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                      <div className="text-xs text-muted-foreground">{s.label}</div>
                      <div className={`text-lg font-mono font-medium ${s.color}`}>{s.value}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4 text-sm text-muted-foreground space-y-2">
                  <div>No feedback data yet.</div>
                  <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => router.push('/chat')}>
                    Start Chatting
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

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
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Conversations ({conversations.length})</CardTitle>
            <Button size="sm" variant="ghost" onClick={handleLoadConversations}>
              <IconRefresh className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Input
                value={newConvName}
                onChange={e => setNewConvName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCreateConversation()}
                placeholder="New conversation name..."
              />
              <Button size="sm" onClick={handleCreateConversation} disabled={creating || !newConvName.trim()}>
                Create
              </Button>
            </div>
            {conversations.length === 0 ? (
              <p className="text-sm text-muted-foreground">No conversations yet.</p>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {conversations.map(conv => (
                  <div key={conv.id} className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm group hover:bg-muted/50 transition-colors">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">{conv.name}</span>
                        {conv.pinned && <span className="text-xs bg-primary/10 text-primary px-1 rounded">pinned</span>}
                        {conv.starred && <span className="text-xs bg-warning/10 text-warning px-1 rounded">starred</span>}
                      </div>
                      <div className="text-xs text-muted-foreground mt-0.5">
                        {conv.message_count} messages · {conv.created_at ? new Date(conv.created_at).toLocaleDateString() : '—'}
                      </div>
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                      <Button size="sm" variant="ghost" onClick={() => handleTogglePin(conv)}>
                        {conv.pinned ? 'Unpin' : 'Pin'}
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => handleToggleStar(conv)}>
                        {conv.starred ? 'Unstar' : 'Star'}
                      </Button>
                      <Button size="sm" variant="ghost" className="text-destructive" onClick={() => handleDeleteConversation(conv.id)}>
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
            <CardHeader>
              <CardTitle className="text-base">Training Data</CardTitle>
            </CardHeader>
            <CardContent>
              {trainStats ? (
                <KpiGrid>
                  <StatCard label="Feedback Pairs" value={String(trainStats.feedback_pairs ?? 0)} />
                  <StatCard label="Last Training" value={trainStats.last_training ? new Date(trainStats.last_training).toLocaleDateString() : 'Never'} />
                  <StatCard label="Quality Score" value={trainStats.quality_score != null ? `${(trainStats.quality_score * 100).toFixed(1)}%` : '—'} />
                </KpiGrid>
              ) : (
                <div className="text-center py-4 text-sm text-muted-foreground">
                  No training data available.
                  <div className="mt-2">
                    <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => router.push('/training')}>
                      Go to Training
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {workflow && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-base">Workflow Controls</CardTitle>
                <Button size="sm" variant="ghost" onClick={handleRefreshStats}>
                  <IconRefresh className="h-4 w-4" />
                </Button>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    { label: 'Status', value: workflow.running ? 'Running' : 'Stopped', color: workflow.running ? 'text-success' : 'text-muted-foreground' },
                    { label: 'Workflow Runs', value: String(workflow.stats?.workflow_runs ?? 0) },
                    { label: 'Aggregations', value: String(workflow.stats?.aggregations_performed ?? 0) },
                    { label: 'Prunes', value: String(workflow.stats?.prunes_performed ?? 0) },
                  ].map(s => (
                    <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                      <div className="text-xs text-muted-foreground">{s.label}</div>
                      <div className={`text-lg font-mono font-medium ${s.color ?? ''}`}>{s.value}</div>
                    </div>
                  ))}
                </div>
                {workflow.stats?.feedback_recorded != null && (
                  <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                    <span>Feedback recorded: {workflow.stats.feedback_recorded}</span>
                    <span>Exports: {workflow.stats.exports_performed ?? 0}</span>
                  </div>
                )}
                <div className="flex gap-2">
                  <Button size="sm" onClick={() => feedbackController.triggerWorkflowAction('aggregate').then(() => { addToast('Aggregation triggered', 'success'); handleRefreshStats() }).catch(() => addToast('Aggregation failed', 'error'))}>
                    Aggregate
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => feedbackController.triggerWorkflowAction('prune').then(() => { addToast('Prune triggered', 'success'); handleRefreshStats() }).catch(() => addToast('Prune failed', 'error'))}>
                    Prune
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => feedbackController.triggerWorkflowAction('export').then(() => { addToast('Export triggered', 'success') }).catch(() => addToast('Export failed', 'error'))}>
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
