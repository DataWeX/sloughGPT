'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button, Input } from '@sloughgpt/strui'
import { Spinner } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { Tabs } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { tokenizerController, type TokenizerStats } from '@/lib/tokenizer-controller'
import { TokenBadge, SegBar, SampleTable } from '@/components/tokenizer/TokenizerComponents'

export default function TokenizerPage() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<TokenizerStats | null>(null)
  const [tab, setTab] = useState<'playground' | 'samples' | 'merges' | 'vocab' | 'decompose' | 'analyze'>('playground')
  const [loading, setLoading] = useState(true)

  // Playground
  const [input, setInput] = useState('')
  const [tokResult, setTokResult] = useState<{ tokens: string[]; ids: number[] } | null>(null)
  const [preResult, setPreResult] = useState<{ pretokens: string[]; segments: { text: string; char_count: number; pct: number }[] } | null>(null)
  const [tokLoading, setTokLoading] = useState(false)

  // Merges
  const [merges, setMerges] = useState<{ index: number; left: string; right: string; token: string }[]>([])
  const [mergeTotal, setMergeTotal] = useState(0)
  const [mergeSearch, setMergeSearch] = useState('')

  // Vocab
  const [vocabEntries, setVocabEntries] = useState<{ id: number; token: string; is_special: boolean }[]>([])
  const [vocabTotal, setVocabTotal] = useState(0)
  const [vocabOffset, setVocabOffset] = useState(0)
  const [vocabSearch, setVocabSearch] = useState('')

  // Decompose
  const [decompToken, setDecompToken] = useState('')
  const [decompResult, setDecompResult] = useState<{
    token: string; id: number; merge_path: { left: string; right: string; into: string }[]; base_chars: string[]
  } | null>(null)
  const [decompLoading, setDecompLoading] = useState(false)

  // Analysis
  const [analysisText, setAnalysisText] = useState('')
  const [analysisResult, setAnalysisResult] = useState<{
    compression_ratio: number; vocab_utilization: number; top_tokens: { token: string; count: number; pct: number }[]
  } | null>(null)

  // Training
  const [training, setTraining] = useState(false)

  const fetchStats = useCallback(async () => {
    try {
      const s = await tokenizerController.getStats()
      setStats(s)
    } catch { addToast('Failed to load stats', 'error') }
  }, [addToast])

  const fetchMerges = useCallback(async (limit = 100) => {
    try {
      const res = await tokenizerController.getMerges(limit)
      setMerges(res.merges)
      setMergeTotal(res.total)
    } catch { addToast('Failed to load merge rules', 'error') }
  }, [addToast])

  const fetchVocab = useCallback(async (limit = 50, offset = 0) => {
    try {
      const res = await tokenizerController.getVocab(limit, offset)
      setVocabEntries(res.entries)
      setVocabTotal(res.total)
      setVocabOffset(res.offset)
    } catch { addToast('Failed to load vocabulary', 'error') }
  }, [addToast])

  useEffect(() => {
    Promise.all([fetchStats(), fetchMerges(), fetchVocab()]).finally(() => setLoading(false))
  }, [fetchStats, fetchMerges, fetchVocab])

  const handleTokenize = async () => {
    if (!input.trim()) return
    setTokLoading(true)
    try {
      const [tok, pre] = await Promise.all([
        tokenizerController.tokenize(input),
        tokenizerController.pretokenize(input),
      ])
      setTokResult(tok)
      setPreResult(pre)
    } catch (e: any) {
      setTokResult({ tokens: [`Error: ${e.message}`], ids: [] })
      setPreResult(null)
    }
    setTokLoading(false)
  }

  const handleDecompose = async () => {
    if (!decompToken.trim()) return
    setDecompLoading(true)
    try {
      const res = await tokenizerController.decomposeToken(decompToken)
      setDecompResult(res)
    } catch (e: any) {
      addToast('Something went wrong decomposing token', 'error')
      setDecompResult(null)
    }
    setDecompLoading(false)
  }

  const handleAnalyze = async () => {
    if (!analysisText.trim()) return
    try {
      const texts = analysisText.split('\n').filter(Boolean)
      const res = await tokenizerController.analyzeCorpus(texts)
      setAnalysisResult(res)
    } catch (e: any) {
      addToast('Something went wrong analyzing text', 'error')
    }
  }

  const filteredMerges = mergeSearch
    ? merges.filter(m => m.token.includes(mergeSearch) || m.left.includes(mergeSearch) || m.right.includes(mergeSearch))
    : merges

  const filteredVocab = vocabSearch
    ? vocabEntries.filter(e => e.token.includes(vocabSearch))
    : vocabEntries

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Tokenizer" />}
        right={
          <Button variant="outline" size="sm" onClick={() => { fetchStats(); fetchMerges(); fetchVocab() }}>
            Refresh
          </Button>
        }
      />
      <div className="space-y-4">

        {/* Stats */}
        <Card>
          <CardHeader><CardTitle className="text-base">Vocabulary</CardTitle></CardHeader>
          <CardContent>
            {loading ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[1,2,3,4].map(i => (
                  <div key={i} className="h-16 animate-pulse bg-muted rounded" />
                ))}
              </div>
            ) : (
              <KpiGrid columns={4}>
                <StatCard label="Vocabulary size" value={stats?.vocab_size ?? '—'} />
                <StatCard label="Base chars" value={stats?.base_chars ?? '—'} />
                <StatCard label="Word pieces" value={stats?.merged_subwords ?? '—'} />
                <StatCard label="Merges" value={stats?.total_merges ?? '—'} />
              </KpiGrid>
            )}
            {stats && (
              <p className="mt-2 text-xs text-muted-foreground">
                {stats.special_tokens} special tokens
              </p>
            )}
          </CardContent>
        </Card>

        {/* Playground */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Playground</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
                placeholder="Type text to tokenize..."
                className="text-sm"
                onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && handleTokenize()}
                aria-label="Text to tokenize"
              />
              <Button size="sm" onClick={handleTokenize} disabled={tokLoading}>
                {tokLoading ? '...' : 'Tokenize'}
              </Button>
            </div>

            {preResult && preResult.segments.length > 1 && (
              <div className="space-y-1.5">
                <p className="text-[11px] font-medium text-muted-foreground">Pre-tokenization ({preResult.pretokens.length} segments)</p>
                <div className="flex flex-wrap gap-1">
                  {preResult.pretokens.map((p, i) => (
                    <span key={i} className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-muted border border-border/40">
                      {p === ' ' ? <span className="text-warning">␣</span> : p}
                    </span>
                  ))}
                </div>
                <Collapsible className="text-[10px] text-muted-foreground">
                  <CollapsibleTrigger className="cursor-pointer hover:text-foreground">Segment size distribution</CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="mt-1 space-y-0.5">
                      {preResult.segments.map((seg, i) => (
                        <SegBar key={i} pct={seg.pct} label={seg.text} />
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            )}

            {tokResult && (
              <div className="space-y-2">
                <p className="text-[11px] font-medium text-muted-foreground">Tokenized ({tokResult.tokens.length} tokens)</p>
                <div className="flex flex-wrap gap-1.5">
                  {tokResult.tokens.map((t, i) => (
                    <TokenBadge key={i} token={t} isSpecial={t.startsWith('<')} />
                  ))}
                </div>
                <Collapsible className="text-xs text-muted-foreground">
                  <CollapsibleTrigger className="cursor-pointer hover:text-foreground">IDs</CollapsibleTrigger>
                  <CollapsibleContent>
                    <p className="mt-1 font-mono text-[10px]">[{tokResult.ids.join(', ')}]</p>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Tab navigation */}
        <Tabs
          value={tab}
          onChange={setTab as any}
          tabs={[
            { value: 'playground', label: 'Playground' },
            { value: 'samples', label: 'Samples' },
            { value: 'merges', label: `Merge rules (${mergeTotal})` },
            { value: 'vocab', label: `Vocabulary (${vocabTotal})` },
            { value: 'decompose', label: 'Decompose' },
            { value: 'analyze', label: 'Analyze' },
          ]}
        />

        {/* Samples tab */}
        {tab === 'samples' && (
          <Card>
            <CardHeader><CardTitle className="text-base">Word Samples</CardTitle></CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground mb-3">Common words and their tokenization breakdown.</p>
              <SampleTable />
            </CardContent>
          </Card>
        )}

        {/* Merges tab */}
        {tab === 'merges' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Merge Rules ({mergeTotal})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={mergeSearch}
                onChange={e => setMergeSearch(e.target.value)}
                placeholder="Search merges..."
                className="text-sm"
                aria-label="Search merge rules"
              />
              <div className="max-h-80 overflow-y-auto space-y-0.5">
                {filteredMerges.map(m => (
                  <div key={m.index} className="flex items-center gap-2 text-xs font-mono py-1 px-2 rounded hover:bg-muted/40">
                    <span className="text-muted-foreground w-8 text-right">#{m.index}</span>
                    <span className="text-foreground">{m.left}</span>
                    <span className="text-muted-foreground">+</span>
                    <span className="text-foreground">{m.right}</span>
                    <span className="text-muted-foreground">→</span>
                    <span className="text-primary font-semibold">{m.token}</span>
                  </div>
                ))}
                {filteredMerges.length === 0 && (
                  <p className="text-xs text-muted-foreground py-4 text-center">No merges match &ldquo;{mergeSearch}&rdquo;</p>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Vocab tab */}
        {tab === 'vocab' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Vocabulary ({vocabTotal} tokens)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={vocabSearch}
                onChange={e => setVocabSearch(e.target.value)}
                placeholder="Search tokens..."
                className="text-sm"
                aria-label="Search vocabulary tokens"
              />
              <div className="max-h-80 overflow-y-auto space-y-0.5">
                {filteredVocab.map(e => (
                  <div key={e.id} className="flex items-center gap-2 text-xs py-1 px-2 rounded hover:bg-muted/40">
                    <span className="text-muted-foreground w-10 text-right tabular-nums">#{e.id}</span>
                    <TokenBadge token={e.token} isSpecial={e.is_special} />
                    {e.is_special && <span className="text-[10px] text-primary ml-1">special</span>}
                  </div>
                ))}
                {filteredVocab.length === 0 && (
                  <p className="text-xs text-muted-foreground py-4 text-center">No tokens match &ldquo;{vocabSearch}&rdquo;</p>
                )}
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-border/20">
                <Button
                  variant="outline" size="sm"
                  disabled={vocabOffset === 0}
                  onClick={() => fetchVocab(50, Math.max(0, vocabOffset - 50))}
                >
                  Previous
                </Button>
                <span className="text-xs text-muted-foreground">
                  {vocabOffset + 1}–{Math.min(vocabOffset + 50, vocabTotal)} of {vocabTotal}
                </span>
                <Button
                  variant="outline" size="sm"
                  disabled={vocabOffset + 50 >= vocabTotal}
                  onClick={() => fetchVocab(50, vocabOffset + 50)}
                >
                  Next
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Decompose tab */}
        {tab === 'decompose' && (
          <Card>
            <CardHeader><CardTitle className="text-base">Token Decomposition</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">Explore how a token is built from its merge tree.</p>
              <div className="flex gap-2">
                <Input
                  value={decompToken}
                  onChange={e => setDecompToken(e.target.value)}
                  placeholder="Enter a token (e.g. 'hello')..."
                  className="text-sm font-mono"
                  onKeyDown={e => e.key === 'Enter' && handleDecompose()}
                  aria-label="Token to decompose"
                />
                <Button size="sm" onClick={handleDecompose} disabled={decompLoading}>
                  {decompLoading ? '...' : 'Trace'}
                </Button>
              </div>
              {decompResult && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground">Token:</span>
                    <TokenBadge token={decompResult.token} />
                    <span className="text-muted-foreground">ID: #{decompResult.id}</span>
                  </div>
                  {decompResult.merge_path.length > 0 && (
                    <div className="space-y-0.5 pl-2 border-l-2 border-primary/20">
                      {decompResult.merge_path.map((step, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs font-mono py-0.5">
                          <span className="text-muted-foreground w-4">{i + 1}.</span>
                          <span>{step.left}</span>
                          <span className="text-muted-foreground">+</span>
                          <span>{step.right}</span>
                          <span className="text-muted-foreground">→</span>
                          <span className="text-primary font-semibold">{step.into}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {decompResult.base_chars.length > 0 && (
                    <div className="text-xs text-muted-foreground">
                      Base chars: {decompResult.base_chars.map(c => c === ' ' ? '␣' : c).join(' ')}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Analyze tab */}
        {tab === 'analyze' && (
          <Card>
            <CardHeader><CardTitle className="text-base">Text Analysis</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">Paste text to see how it gets split into tokens.</p>
              <textarea
                value={analysisText}
                onChange={e => setAnalysisText(e.target.value)}
                placeholder={`the quick brown fox jumps over the lazy dog\nhello world\nmachine learning transformers`}
                className="w-full h-24 text-xs font-mono p-2 rounded border border-border/60 bg-background resize-none"
                aria-label="Corpus text for analysis"
              />
              <Button size="sm" onClick={handleAnalyze} disabled={!analysisText.trim()}>Analyze</Button>
              {analysisResult && (
                <div className="space-y-3">
                  <KpiGrid columns={3}>
                    <StatCard label="Compression" value={analysisResult.compression_ratio.toFixed(2)} />
                    <StatCard label="Vocabulary usage" value={`${(analysisResult.vocab_utilization * 100).toFixed(1)}%`} />
                    <StatCard label="Top tokens shown" value={analysisResult.top_tokens.length} />
                  </KpiGrid>
                  <div>
                    <p className="text-[11px] font-medium text-muted-foreground mb-1">Most frequent tokens</p>
                    <div className="space-y-0.5">
                      {analysisResult.top_tokens.map((t, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs py-0.5">
                          <span className="text-muted-foreground w-6 text-right">{i + 1}.</span>
                          <TokenBadge token={t.token} />
                          <span className="text-muted-foreground">×{t.count}</span>
                          <span className="text-[10px] text-muted-foreground">({(t.pct * 100).toFixed(1)}%)</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Train */}
        <Card>
          <CardHeader><CardTitle className="text-base">Train</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Train a new tokenizer on the Tiny Shakespeare dataset (~2K lines).
            </p>
            <Button
              size="sm"
              onClick={async () => {
                setTraining(true)
                try {
                  await tokenizerController.trainTokenizer()
                  addToast('Tokenizer trained on Shakespeare', 'success')
                  await Promise.all([fetchStats(), fetchMerges(), fetchVocab()])
                } catch (e: any) {
                  addToast('Something went wrong training the tokenizer', 'error')
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
