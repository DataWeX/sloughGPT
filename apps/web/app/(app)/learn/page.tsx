'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Textarea, cn } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { learnerController, type LearnerStatus } from '@/lib/learner-controller'
import { LearningInsightsCard } from '@/components/learn/LearningInsightsCard'
import { useToastStore } from '@/lib/toast-store'

type Tab = 'search' | 'ingest' | 'knowledge' | 'feeds' | 'train' | 'evaluate' | 'deploy'

export default function LearnPage() {
  const [status, setStatus] = useState<LearnerStatus | null>(null)
  const [tab, setTab] = useState<Tab>('search')
  const [loading, setLoading] = useState(true)

  const [searchQuery, setSearchQuery] = useState('')
  const [searchResult, setSearchResult] = useState<string | null>(null)
  const [searching, setSearching] = useState(false)

  const [ingestUrl, setIngestUrl] = useState('')
  const [ingestText, setIngestText] = useState('')
  const [ingestResult, setIngestResult] = useState<string | null>(null)
  const [ingesting, setIngesting] = useState(false)

  const [knowledge, setKnowledge] = useState<Array<{ content: string; topic: string; source: string; importance: number }>>([])
  const [knowledgeQuery, setKnowledgeQuery] = useState('')
  const [loadingKnowledge, setLoadingKnowledge] = useState(false)

  const [feeds, setFeeds] = useState<Array<{ url: string; interval: number; last_poll?: string }>>([])
  const [newFeedUrl, setNewFeedUrl] = useState('')
  const [feedMsg, setFeedMsg] = useState<string | null>(null)
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState<{ status: string; loss?: number } | null>(null)
  const [evaluating, setEvaluating] = useState(false)
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null)
  const [deploying, setDeploying] = useState(false)
  const [deployResult, setDeployResult] = useState<string | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const handleTrain = async () => {
    setTraining(true)
    setTrainResult(null)
    try {
      const result = await learnerController.train()
      setTrainResult(result)
      addToast('Training complete', 'success')
    } catch (e) {
      addToast(`Training failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setTraining(false)
    }
  }

  const handleEvaluate = async () => {
    setEvaluating(true)
    setEvalResult(null)
    try {
      const result = await learnerController.evaluate()
      setEvalResult(result.metrics)
      addToast('Evaluation complete', 'success')
    } catch (e) {
      addToast(`Evaluation failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setEvaluating(false)
    }
  }

  const handleDeploy = async () => {
    setDeploying(true)
    setDeployResult(null)
    try {
      const result = await learnerController.deploy()
      setDeployResult(result.status)
      addToast('Deploy complete', 'success')
    } catch (e) {
      addToast(`Deploy failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setDeploying(false)
    }
  }

  useEffect(() => {
    learnerController.status().then(s => {
      setStatus(s)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    setSearchResult(null)
    try {
      const res = await learnerController.search(searchQuery)
      setSearchResult(`Ingested ${res.tokens_ingested} tokens, ${res.new_facts} new facts`)
    } catch (err) {
      setSearchResult(err instanceof Error ? err.message : 'Could not search')
    } finally {
      setSearching(false)
    }
  }

  const handleIngestUrl = async () => {
    if (!ingestUrl.trim()) return
    setIngesting(true)
    setIngestResult(null)
    try {
      const res = await learnerController.ingestUrl(ingestUrl)
      setIngestResult(`Added ${res.facts_added} facts from URL`)
      setIngestUrl('')
    } catch (err) {
      setIngestResult(err instanceof Error ? err.message : 'Could not ingest')
    } finally {
      setIngesting(false)
    }
  }

  const handleIngestText = async () => {
    if (!ingestText.trim()) return
    setIngesting(true)
    setIngestResult(null)
    try {
      const res = await learnerController.ingestText(ingestText)
      setIngestResult(`Added ${res.facts_added} facts from text`)
      setIngestText('')
    } catch (err) {
      setIngestResult(err instanceof Error ? err.message : 'Could not ingest')
    } finally {
      setIngesting(false)
    }
  }

  const handleLoadKnowledge = async () => {
    setLoadingKnowledge(true)
    try {
      const res = await learnerController.queryKnowledge(knowledgeQuery || undefined)
      setKnowledge(res.facts ?? [])
    } catch {
      addToast('Could not load knowledge', 'error')
    } finally {
      setLoadingKnowledge(false)
    }
  }

  const handleLoadFeeds = async () => {
    try {
      const res = await learnerController.listFeeds()
      setFeeds(res.feeds ?? [])
    } catch {
      addToast('Could not load feeds', 'error')
    }
  }

  const handleSubscribeFeed = async () => {
    if (!newFeedUrl.trim()) return
    try {
      await learnerController.subscribeFeed(newFeedUrl)
      setFeedMsg('Subscribed')
      setNewFeedUrl('')
      await handleLoadFeeds()
    } catch { setFeedMsg('Failed') }
  }

  const toolbar = (
    <div className="flex gap-1 border-b border-border/30 pb-0">
      {(['search', 'ingest', 'knowledge', 'feeds'] as Tab[]).map(t => (
        <button
          key={t}
          type="button"
          onClick={() => {
            setTab(t)
            if (t === 'knowledge') handleLoadKnowledge()
            if (t === 'feeds') handleLoadFeeds()
          }}
          className={cn('px-3 py-1.5 text-xs font-medium rounded-t transition-colors', tab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground')}
        >
          {t.charAt(0).toUpperCase() + t.slice(1)}
        </button>
      ))}
    </div>
  )

  return (
    <PageContainer
      title="Learner"
      subtitle={status ? `${status.total_tokens_ingested} tokens · ${status.feeds_subscribed} feeds` : 'Continual web learning'}
      loading={loading}
      toolbar={toolbar}
    >
      {status && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Tokens Ingested', value: status.total_tokens_ingested },
            { label: 'Train Steps', value: status.train_steps_completed },
            { label: 'Feeds', value: status.feeds_subscribed },
            { label: 'Buffer', value: `${status.buffer_size}/${status.buffer_capacity}` },
          ].map(s => (
            <div key={s.label} className="rounded-md bg-muted/30 p-4 text-center">
              <div className="text-xs text-muted-foreground">{s.label}</div>
              <div className="text-base font-mono font-medium">{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {knowledge.length > 0 && <LearningInsightsCard facts={knowledge} />}

      {tab === 'search' && (
        <Card>
          <CardContent className="pt-4 space-y-3">
            <p className="text-sm text-muted-foreground">Search the web, fetch articles, and learn from them.</p>
            {searchResult && (
              <div className="rounded-md bg-primary/10 border border-primary/20 px-3 py-2 text-sm text-primary">{searchResult}</div>
            )}
            <div className="flex gap-2">
              <Input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
                placeholder="Search query..."
              />
              <Button size="sm" onClick={handleSearch} disabled={searching || !searchQuery.trim()}>
                {searching ? 'Searching...' : 'Search & Learn'}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {tab === 'ingest' && (
        <Card>
          <CardContent className="pt-4 space-y-3">
            {ingestResult && (
              <div className="rounded-md bg-primary/10 border border-primary/20 px-3 py-2 text-sm text-primary">{ingestResult}</div>
            )}
            <div>
              <div className="text-xs text-muted-foreground mb-1">From URL</div>
              <div className="flex gap-2">
                <Input value={ingestUrl} onChange={e => setIngestUrl(e.target.value)} placeholder="https://..." />
                <Button size="sm" onClick={handleIngestUrl} disabled={ingesting || !ingestUrl.trim()}>Ingest URL</Button>
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">From Text</div>
              <Textarea value={ingestText} onChange={e => setIngestText(e.target.value)} placeholder="Paste text to learn from..." rows={4} />
              <Button size="sm" className="mt-1" onClick={handleIngestText} disabled={ingesting || !ingestText.trim()}>Ingest Text</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {tab === 'knowledge' && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Knowledge ({knowledge.length})</CardTitle>
            <Button size="sm" variant="ghost" onClick={handleLoadKnowledge} aria-label="Refresh knowledge">
              <IconRefresh className={cn('h-4 w-4', loadingKnowledge && 'animate-spin')} />
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Input value={knowledgeQuery} onChange={e => setKnowledgeQuery(e.target.value)} placeholder="Filter by topic..." />
              <Button size="sm" variant="outline" onClick={handleLoadKnowledge}>Search</Button>
            </div>
            {knowledge.length === 0 ? (
              <p className="text-sm text-muted-foreground">No knowledge yet. Use Search or Ingest to learn.</p>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {knowledge.map((f, i) => (
                  <div key={i} className="rounded-md border border-border/60 px-3 py-2 text-sm">
                    <div className="truncate">{f.content}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {f.topic && <span className="bg-muted px-1 rounded mr-1">{f.topic}</span>}
                      {f.source}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'feeds' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">RSS Feeds</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {feedMsg && (
              <div className="text-sm text-primary">{feedMsg}</div>
            )}
            <div className="flex gap-2">
              <Input value={newFeedUrl} onChange={e => setNewFeedUrl(e.target.value)} placeholder="RSS feed URL..." />
              <Button size="sm" onClick={handleSubscribeFeed}>Subscribe</Button>
            </div>
            {feeds.length === 0 ? (
              <p className="text-sm text-muted-foreground">No feeds subscribed.</p>
            ) : (
              <div className="space-y-1">
                {feeds.map((f, i) => (
                  <div key={i} className="flex items-center justify-between text-xs py-1 border-b border-border/20">
                    <span className="truncate">{f.url}</span>
                    <span className="text-muted-foreground shrink-0 ml-2">{f.interval}s</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'train' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Train Knowledge Model</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">Train a model on ingested knowledge to improve retrieval quality and generate better embeddings.</p>
            <Button onClick={() => void handleTrain()} disabled={training} className="w-full">
              {training ? 'Training...' : 'Start Training'}
            </Button>
            {trainResult && (
              <div className="rounded bg-muted/30 p-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <span className="font-medium">{trainResult.status}</span>
                </div>
                {trainResult.loss != null && (
                  <div className="flex justify-between mt-1">
                    <span className="text-muted-foreground">Loss</span>
                    <span className="font-mono">{trainResult.loss.toFixed(4)}</span>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'evaluate' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Evaluate Knowledge Model</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">Run evaluation metrics on the trained knowledge model to measure retrieval accuracy and quality.</p>
            <Button onClick={() => void handleEvaluate()} disabled={evaluating} className="w-full">
              {evaluating ? 'Evaluating...' : 'Run Evaluation'}
            </Button>
            {evalResult && (
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(evalResult).map(([key, value]) => (
                  <div key={key} className="rounded bg-muted/30 p-2 text-center">
                    <div className="text-xs text-muted-foreground">{key}</div>
                    <div className="text-xs font-mono font-medium">{String(value)}</div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'deploy' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Deploy Knowledge Model</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">Deploy the trained knowledge model to make it available for inference and retrieval.</p>
            <Button onClick={() => void handleDeploy()} disabled={deploying} className="w-full">
              {deploying ? 'Deploying...' : 'Deploy Model'}
            </Button>
            {deployResult && (
              <div className="rounded bg-muted/30 p-3 text-sm">
                <span className="text-muted-foreground">Status: </span>
                <span className="font-medium">{deployResult}</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </PageContainer>
  )
}
