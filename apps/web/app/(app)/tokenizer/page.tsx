'use client'

import { useState, useEffect, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui'
import { StatCard, KpiGrid } from '@/components/ui/display'
import { useToastStore } from '@/lib/toast-store'
import { tokenizerController, type TokenizerStats, type SampleWord } from '@/lib/tokenizer-controller'

function TokenBadge({ token, isSpecial }: { token: string; isSpecial?: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono leading-tight border ${
      isSpecial
        ? 'bg-primary/10 border-primary/30 text-primary'
        : 'bg-card border-border/60 text-foreground'
    }`}>
      {token === ' ' ? <span className="text-warning">␣</span> : token === '\n' ? <span className="text-warning">↵</span> : token}
    </span>
  )
}

function SampleRow({ sample }: { sample: SampleWord }) {
  return (
    <div className="flex items-center gap-3 py-1.5 text-sm border-b border-border/20 last:border-0">
      <span className="w-20 font-mono text-xs text-muted-foreground truncate shrink-0">{sample.word}</span>
      <span className="w-6 text-[10px] text-muted-foreground text-center shrink-0">{sample.count}</span>
      <div className="flex flex-wrap gap-1">
        {sample.tokens.map((t, i) => (
          <TokenBadge key={i} token={t} />
        ))}
      </div>
      <span className="text-[10px] text-muted-foreground ml-auto font-mono">
        [{sample.ids.join(', ')}]
      </span>
    </div>
  )
}

export default function TokenizerPage() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<TokenizerStats | null>(null)
  const [samples, setSamples] = useState<SampleWord[]>([])
  const [tokenizeInput, setTokenizeInput] = useState('')
  const [tokenizeResult, setTokenizeResult] = useState<{ tokens: string[]; ids: number[] } | null>(null)
  const [loading, setLoading] = useState(false)
  const [tab, setTab] = useState<'samples' | 'playground'>('samples')
  const [training, setTraining] = useState(false)

  const fetchStats = useCallback(async () => {
    try {
      const s = await tokenizerController.getStats()
      setStats(s)
    } catch { /* tokenizer might not be trained */ }
  }, [])

  const fetchSamples = useCallback(async () => {
    try {
      const res = await tokenizerController.getSamples()
      setSamples(res.samples)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    fetchStats()
    fetchSamples()
  }, [fetchStats, fetchSamples])

  const handleTokenize = async () => {
    if (!tokenizeInput.trim()) return
    setLoading(true)
    try {
      const res = await tokenizerController.tokenize(tokenizeInput)
      setTokenizeResult(res)
    } catch (e: any) {
      setTokenizeResult({ tokens: [`Error: ${e.message}`], ids: [] })
    }
    setLoading(false)
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Tokenizer" />}
        right={
          <Button variant="outline" size="sm" onClick={() => { fetchStats(); fetchSamples() }}>
            Refresh
          </Button>
        }
      />
      <div className="space-y-4">
        {/* Stats */}
        <Card>
          <CardHeader><CardTitle className="text-base">Vocabulary</CardTitle></CardHeader>
          <CardContent>
            <KpiGrid columns={4}>
              <StatCard label="Vocab size" value={stats?.vocab_size ?? '—'} />
              <StatCard label="Base chars" value={stats?.base_chars ?? '—'} />
              <StatCard label="Subwords" value={stats?.merged_subwords ?? '—'} />
              <StatCard label="Merges" value={stats?.total_merges ?? '—'} />
            </KpiGrid>
            {stats && (
              <p className="mt-3 text-xs text-muted-foreground">
                {stats.special_tokens} special tokens
              </p>
            )}
          </CardContent>
        </Card>

        {/* Tabs */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Tokenization</CardTitle>
            <div className="flex gap-1">
              <Button variant={tab === 'samples' ? 'default' : 'outline'} size="sm" onClick={() => setTab('samples')}>
                Samples
              </Button>
              <Button variant={tab === 'playground' ? 'default' : 'outline'} size="sm" onClick={() => setTab('playground')}>
                Playground
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {tab === 'samples' ? (
              <div>
                <div className="flex items-center gap-3 py-1.5 text-[10px] font-medium text-muted-foreground border-b border-border/40 uppercase tracking-wider">
                  <span className="w-20">Word</span>
                  <span className="w-6 text-center">#</span>
                  <span>Tokens</span>
                  <span className="ml-auto">IDs</span>
                </div>
                <div className="divide-y divide-border/10">
                  {samples.map((s, i) => (
                    <SampleRow key={i} sample={s} />
                  ))}
                  {samples.length === 0 && (
                    <p className="text-xs text-muted-foreground py-4 text-center">
                      No samples available — tokenizer may not be trained
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex gap-2">
                  <Input
                    value={tokenizeInput}
                    onChange={e => setTokenizeInput(e.target.value)}
                    placeholder="Type text to tokenize..."
                    className="text-sm"
                    onKeyDown={e => e.key === 'Enter' && handleTokenize()}
                  />
                  <Button size="sm" onClick={handleTokenize} disabled={loading}>
                    {loading ? '...' : 'Tokenize'}
                  </Button>
                </div>
                {tokenizeResult && (
                  <div className="space-y-2">
                    <div className="flex flex-wrap gap-1.5">
                      {tokenizeResult.tokens.map((t, i) => (
                        <TokenBadge key={i} token={t} isSpecial={t.startsWith('<')} />
                      ))}
                    </div>
                    <details className="text-xs text-muted-foreground">
                      <summary className="cursor-pointer hover:text-foreground">IDs</summary>
                      <p className="mt-1 font-mono">[{tokenizeResult.ids.join(', ')}]</p>
                    </details>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Train */}
        <Card>
          <CardHeader><CardTitle className="text-base">Train</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Train a new BPE tokenizer on the Tiny Shakespeare dataset.
            </p>
            <Button
              size="sm"
              onClick={async () => {
                setTraining(true)
                try {
                  await tokenizerController.trainShakespeare()
                  addToast('Tokenizer trained on Shakespeare', 'success')
                  await Promise.all([fetchStats(), fetchSamples()])
                } catch (e: any) {
                  addToast(e.message || 'Training failed', 'error')
                }
                setTraining(false)
              }}
              disabled={training}
            >
              {training ? <><Spinner className="w-3 h-3 mr-1" /> Training...</> : 'Train on Shakespeare'}
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
