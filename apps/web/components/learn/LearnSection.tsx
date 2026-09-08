'use client'

import { useState, useEffect } from 'react'
import { cn, ActionCard, Card, CardHeader, CardTitle, CardContent, Button, Input, Textarea, Spinner } from '@sloughgpt/strui'
import { SectionLabel } from '@/components/composed/SectionLabel'
import { learnerController, type LearnerStatus } from '@/lib/learner-controller'
import { LearningInsightsCard } from '@/components/learn/LearningInsightsCard'
import { useToastStore } from '@/lib/toast-store'

type Tab = 'search' | 'ingest' | 'knowledge' | 'feeds'

export function LearnSection() {
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
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    let active = true
    learnerController.status().then(s => {
      if (active) {
        setStatus(s)
        setLoading(false)
      }
    }).catch(() => {
      if (active) {
        addToast('Failed to load learner status', 'error')
        setLoading(false)
      }
    })
    return () => { active = false }
  }, [addToast])

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

  return (
    <>
      <div className="flex items-center justify-between border-b border-border/30 pb-1.5 pt-0.5">
        <SectionLabel>Continual Learning</SectionLabel>
        <span className="text-[10px] text-muted-foreground/60 tabular-nums">
          {status ? `${status.total_tokens_ingested} tokens · ${status.feeds_subscribed} feeds` : 'Search the web and learn'}
        </span>
      </div>

      {loading ? (
        <Card><CardContent><div className="h-28 animate-pulse bg-muted/50 rounded-lg" /></CardContent></Card>
      ) : (
        status && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[
              { label: 'Tokens Ingested', value: status.total_tokens_ingested },
              { label: 'Train Steps', value: status.train_steps_completed },
              { label: 'Feeds', value: status.feeds_subscribed },
              { label: 'Buffer', value: `${status.buffer_size}/${status.buffer_capacity}` },
            ].map(s => (
              <div key={s.label} className="rounded-lg bg-muted/30 p-2.5 text-center">
                <div className="text-[10px] text-muted-foreground/60">{s.label}</div>
                <div className="text-[13px] font-mono font-medium tabular-nums">{s.value}</div>
              </div>
            ))}
          </div>
        )
      )}

      {knowledge.length > 0 && <LearningInsightsCard facts={knowledge} />}

      <div className="flex gap-0.5 border-b border-border/30 pb-0">
        {(['search', 'ingest', 'knowledge', 'feeds'] as Tab[]).map(t => (
          <button
            key={t}
            type="button"
            onClick={() => {
              setTab(t)
              if (t === 'knowledge') handleLoadKnowledge()
              if (t === 'feeds') handleLoadFeeds()
            }}
            className={cn('px-2.5 py-1 text-[10px] font-medium rounded-t transition-colors', tab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground/60 hover:text-foreground')}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'search' && (
        <Card>
          <CardContent className="pt-3 space-y-2">
            <p className="text-[11px] text-muted-foreground/60">Search the web, fetch articles, and learn from them.</p>
            {searchResult && (
              <div className="rounded-lg bg-primary/10 border border-primary/20 px-2.5 py-1.5 text-[11px] text-primary">{searchResult}</div>
            )}
            <div className="flex gap-1.5">
              <Input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
                placeholder="Search query..."
                className="h-6 text-[10px]"
              />
              <Button size="sm" className="h-6 text-[10px]" onClick={handleSearch} disabled={searching || !searchQuery.trim()}>
                {searching ? 'Searching...' : 'Search & Learn'}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {tab === 'ingest' && (
        <Card>
          <CardContent className="pt-3 space-y-2">
            {ingestResult && (
              <div className="rounded-lg bg-primary/10 border border-primary/20 px-2.5 py-1.5 text-[11px] text-primary">{ingestResult}</div>
            )}
            <div>
              <div className="text-[10px] text-muted-foreground/60 mb-1">From URL</div>
              <div className="flex gap-1.5">
                <Input value={ingestUrl} onChange={e => setIngestUrl(e.target.value)} placeholder="https://..." className="h-6 text-[10px]" />
                <Button size="sm" className="h-6 text-[10px]" onClick={handleIngestUrl} disabled={ingesting || !ingestUrl.trim()}>Ingest URL</Button>
              </div>
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground/60 mb-1">From Text</div>
              <Textarea value={ingestText} onChange={e => setIngestText(e.target.value)} placeholder="Paste text to learn from..." rows={3} className="text-[11px]" />
              <Button size="sm" className="h-6 text-[10px] mt-1" onClick={handleIngestText} disabled={ingesting || !ingestText.trim()}>Ingest Text</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {tab === 'knowledge' && (
        <ActionCard
          title={`Knowledge (${knowledge.length})`}
          actions={
            <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={handleLoadKnowledge} aria-label="Refresh knowledge">
              <Spinner className="h-3 w-3" />
            </Button>
          }
          contentClassName="space-y-2"
        >
            <div className="flex gap-1.5">
              <Input value={knowledgeQuery} onChange={e => setKnowledgeQuery(e.target.value)} placeholder="Filter by topic..." className="h-6 text-[10px]" />
              <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={handleLoadKnowledge}>Search</Button>
            </div>
            {knowledge.length === 0 ? (
              <p className="text-[11px] text-muted-foreground/60">No knowledge yet. Use Search or Ingest to learn.</p>
            ) : (
              <div className="space-y-1 max-h-80 overflow-y-auto">
                {knowledge.map((f, i) => (
                  <div key={i} className="rounded-lg border border-border/40 px-2.5 py-1.5 text-[10px]">
                    <div className="truncate">{f.content}</div>
                    <div className="text-[9px] text-muted-foreground/60 mt-0.5">
                      {f.topic && <span className="bg-muted/50 px-1 rounded mr-0.5">{f.topic}</span>}
                      {f.source}
                    </div>
                  </div>
                ))}
              </div>
            )}
        </ActionCard>
      )}

      {tab === 'feeds' && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs">RSS Feeds</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {feedMsg && (
              <div className="text-[11px] text-primary">{feedMsg}</div>
            )}
            <div className="flex gap-1.5">
              <Input value={newFeedUrl} onChange={e => setNewFeedUrl(e.target.value)} placeholder="RSS feed URL..." className="h-6 text-[10px]" />
              <Button size="sm" className="h-6 text-[10px]" onClick={handleSubscribeFeed}>Subscribe</Button>
            </div>
            {feeds.length === 0 ? (
              <p className="text-[11px] text-muted-foreground/60">No feeds subscribed.</p>
            ) : (
              <div className="space-y-0.5">
                {feeds.map((f, i) => (
                  <div key={i} className="flex items-center justify-between text-[10px] py-1 border-b border-border/20">
                    <span className="truncate">{f.url}</span>
                    <span className="text-muted-foreground/60 shrink-0 ml-1.5 tabular-nums">{f.interval}s</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </>
  )
}
