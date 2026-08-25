'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { useToastStore } from '@/lib/toast-store'
import {
  tokenTreeController,
  type TokenTreeStats,
  type VocabEntry,
  type MergeRule,
  type SavedTree,
  type SimilarResult,
  type EmbeddingInfo,
  type MatrixSummary,
  type CompareResult,
} from '@/lib/token-tree-controller'

type Tab = 'stats' | 'vocab' | 'merges' | 'similar' | 'embed' | 'saved' | 'matrix' | 'compare'

export default function TokenTreePage() {
  const addToast = useToastStore(s => s.addToast)
  const [tab, setTab] = useState<Tab>('stats')
  const [stats, setStats] = useState<TokenTreeStats | null>(null)
  const [loading, setLoading] = useState(false)

  const [vocabEntries, setVocabEntries] = useState<VocabEntry[]>([])
  const [vocabOffset, setVocabOffset] = useState(0)
  const [vocabTotal, setVocabTotal] = useState(0)

  const [merges, setMerges] = useState<MergeRule[]>([])
  const [mergeQuery, setMergeQuery] = useState('')

  const [savedTrees, setSavedTrees] = useState<SavedTree[]>([])
  const [saveName, setSaveName] = useState('')

  const [similarQuery, setSimilarQuery] = useState('')
  const [similarResult, setSimilarResult] = useState<SimilarResult | null>(null)

  const [embedQuery, setEmbedQuery] = useState('')
  const [embedResult, setEmbedResult] = useState<EmbeddingInfo | null>(null)

  const [matrixResult, setMatrixResult] = useState<MatrixSummary | null>(null)

  const [compareA, setCompareA] = useState('')
  const [compareB, setCompareB] = useState('')
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null)

  const [trainTexts, setTrainTexts] = useState('')
  const [vocabSize, setVocabSize] = useState(256)

  const loadStats = useCallback(async () => {
    try {
      const s = await tokenTreeController.getStats()
      setStats(s)
    } catch { /* silent */ }
  }, [])

  useEffect(() => { void loadStats() }, [loadStats])

  const handleTrain = async () => {
    setLoading(true)
    try {
      const texts = trainTexts.split('\n').filter(t => t.trim())
      const result = await tokenTreeController.train({ texts: texts.length > 0 ? texts : undefined, vocab_size: vocabSize })
      addToast(`Trained — ${result.vocab_size} vocab, ${result.embedding_points} embeddings`, 'success')
      void loadStats()
    } catch (e) {
      addToast(`Training failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const loadVocab = useCallback(async (offset: number) => {
    setLoading(true)
    try {
      const result = await tokenTreeController.getVocab(50, offset)
      setVocabEntries(result.entries)
      setVocabTotal(result.total)
      setVocabOffset(offset)
    } catch (e) {
      addToast(`Could not load vocab: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  const loadMerges = useCallback(async (query: string) => {
    setLoading(true)
    try {
      const result = await tokenTreeController.getMerges(30, query)
      setMerges(result)
    } catch (e) {
      addToast(`Could not load merges: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  const loadSaved = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tokenTreeController.listSaved()
      setSavedTrees(result)
    } catch (e) {
      addToast(`Could not load saved trees: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])


  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void loadStats() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [loadStats])
  useEffect(() => {
    if (tab === 'vocab') void loadVocab(0)
    if (tab === 'merges') void loadMerges('')
    if (tab === 'saved') void loadSaved()
    if (tab === 'matrix') {
      setLoading(true)
      tokenTreeController.getMatrixSummary(8).then(r => setMatrixResult(r)).catch(() => {}).finally(() => setLoading(false))
    }
  }, [tab, loadVocab, loadMerges, loadSaved])

  const handleSave = async () => {
    if (!saveName.trim()) return
    try {
      await tokenTreeController.saveTree(saveName)
      addToast(`Saved as "${saveName}"`, 'success')
      setSaveName('')
      void loadSaved()
    } catch (e) {
      addToast(`Could not save: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    }
  }

  const handleLoad = async (name: string) => {
    try {
      await tokenTreeController.loadTree(name)
      addToast(`Loaded "${name}"`, 'success')
      void loadStats()
    } catch (e) {
      addToast(`Could not load: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    }
  }

  const handleDeleteTree = async (name: string) => {
    try {
      await tokenTreeController.deleteSavedTree(name)
      addToast(`Deleted "${name}"`, 'success')
      void loadSaved()
    } catch (e) {
      addToast(`Could not delete: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    }
  }

  const handleSimilar = async () => {
    if (!similarQuery.trim()) return
    setLoading(true)
    setSimilarResult(null)
    try {
      const result = await tokenTreeController.similar(similarQuery, 10)
      setSimilarResult(result)
    } catch (e) {
      addToast(`Similar search failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleEmbed = async () => {
    if (!embedQuery.trim()) return
    setLoading(true)
    setEmbedResult(null)
    try {
      const result = await tokenTreeController.getEmbedding(embedQuery, 8)
      setEmbedResult(result)
    } catch (e) {
      addToast(`Embedding failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleCompare = async () => {
    if (!compareA.trim() || !compareB.trim()) return
    setLoading(true)
    setCompareResult(null)
    try {
      const result = await tokenTreeController.compare(compareA, compareB, 10)
      setCompareResult(result)
    } catch (e) {
      addToast(`Compare failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const tabs: { key: Tab; label: string }[] = [
    { key: 'stats', label: 'Stats' },
    { key: 'vocab', label: 'Vocabulary' },
    { key: 'merges', label: 'Merges' },
    { key: 'similar', label: 'Similar' },
    { key: 'embed', label: 'Embedding' },
    { key: 'saved', label: 'Saved' },
    { key: 'matrix', label: 'Matrix' },
    { key: 'compare', label: 'Compare' },
  ]



  return (
    <PageContainer
      title="Token Tree"
      subtitle="BPE merge tree with learned embeddings"
      headerRight={
        <Button size="sm" variant="ghost" onClick={() => void loadStats()}>Refresh</Button>
      }
    >
      <div className="space-y-4">
        {stats && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: 'Trained', value: stats.trained ? 'Yes' : 'No' },
              { label: 'Vocab Size', value: stats.vocab_size.toLocaleString() },
              { label: 'Merges', value: stats.num_merges.toLocaleString() },
              { label: 'Embeddings', value: stats.embedding_points.toLocaleString() },
              { label: 'Base Tokens', value: stats.num_base_tokens.toLocaleString() },
              { label: 'Compression', value: stats.embedding_compression_ratio.toFixed(2) },
              { label: 'Embed Dim', value: stats.embed_dim },
            ].map(s => (
              <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                <div className="text-[10px] text-muted-foreground">{s.label}</div>
                <div className="text-sm font-mono font-medium">{s.value}</div>
              </div>
            ))}
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Train Tree</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label className="text-xs">Vocab Size</Label>
                <Input type="number" value={vocabSize} onChange={e => setVocabSize(Number(e.target.value))} className="h-8 text-xs" />
              </div>
              <div className="space-y-1">
                <Label htmlFor="train-texts" className="text-xs">Training Texts (one per line, optional)</Label>
                <textarea id="train-texts" value={trainTexts} onChange={e => setTrainTexts(e.target.value)} rows={3} className="w-full rounded-md border border-input bg-background p-2 text-xs font-mono" placeholder="Leave empty to use default corpus..." />
              </div>
            </div>
            <Button onClick={() => void handleTrain()} disabled={loading} className="w-full">
              {loading ? 'Training...' : 'Train Token Tree'}
            </Button>
          </CardContent>
        </Card>

        <div className="flex gap-1 rounded-lg border border-border bg-muted/30 p-1">
          {tabs.map(t => (
            <button
              type="button"
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                tab === t.key ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === 'vocab' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Vocabulary ({vocabTotal} tokens)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="max-h-[400px] overflow-y-auto">
                <table className="w-full text-xs" aria-label="Vocabulary table">
                  <thead>
                    <tr className="border-b border-border text-left text-muted-foreground">
                      <th scope="col" className="pb-1.5 font-medium">ID</th>
                      <th scope="col" className="pb-1.5 font-medium">Token</th>
                      <th scope="col" className="pb-1.5 font-medium">Freq</th>
                      <th scope="col" className="pb-1.5 font-medium">Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vocabEntries.map(e => (
                      <tr key={e.id} className="border-b border-border/50">
                        <td className="py-1 font-mono text-muted-foreground">{e.id}</td>
                        <td className="py-1 font-mono">{e.token}</td>
                        <td className="py-1">{e.freq}</td>
                        <td className="py-1">
                          {e.is_special && <span className="rounded bg-yellow-500/10 px-1 py-0.5 text-[10px] text-yellow-600">special</span>}
                          {e.is_merged && <span className="rounded bg-blue-500/10 px-1 py-0.5 text-[10px] text-blue-600">merged</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex justify-center gap-2">
                <Button size="sm" variant="outline" disabled={vocabOffset === 0} onClick={() => void loadVocab(vocabOffset - 50)}>Previous</Button>
                <span className="text-xs text-muted-foreground self-center">{vocabOffset + 1}-{Math.min(vocabOffset + 50, vocabTotal)} of {vocabTotal}</span>
                <Button size="sm" variant="outline" disabled={vocabOffset + 50 >= vocabTotal} onClick={() => void loadVocab(vocabOffset + 50)}>Next</Button>
              </div>
            </CardContent>
          </Card>
        )}

        {tab === 'merges' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">BPE Merge Rules</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input value={mergeQuery} onChange={e => setMergeQuery(e.target.value)} placeholder="Filter merges..." className="h-8 text-xs" onKeyDown={e => e.key === 'Enter' && void loadMerges(mergeQuery)} />
                <Button onClick={() => void loadMerges(mergeQuery)} disabled={loading} className="shrink-0">Search</Button>
              </div>
              <div className="max-h-[400px] overflow-y-auto">
                <table className="w-full text-xs" aria-label="BPE merge rules table">
                  <thead>
                    <tr className="border-b border-border text-left text-muted-foreground">
                      <th scope="col" className="pb-1.5 font-medium">Rank</th>
                      <th scope="col" className="pb-1.5 font-medium">Left</th>
                      <th scope="col" className="pb-1.5 font-medium">Right</th>
                      <th scope="col" className="pb-1.5 font-medium">Result</th>
                      <th scope="col" className="pb-1.5 font-medium">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {merges.map(m => (
                      <tr key={m.rank} className="border-b border-border/50">
                        <td className="py-1 font-mono text-muted-foreground">{m.rank}</td>
                        <td className="py-1 font-mono">{m.left}</td>
                        <td className="py-1 font-mono">{m.right}</td>
                        <td className="py-1 font-mono text-primary">{m.token}</td>
                        <td className="py-1">{m.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {tab === 'similar' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Find Similar Tokens</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input value={similarQuery} onChange={e => setSimilarQuery(e.target.value)} placeholder="Enter a token..." className="h-8 text-xs" onKeyDown={e => e.key === 'Enter' && void handleSimilar()} />
                <Button onClick={() => void handleSimilar()} disabled={loading} className="shrink-0">Find</Button>
              </div>
              {similarResult && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">Results for &quot;{similarResult.query}&quot;</p>
                  <div className="space-y-1">
                    {similarResult.neighbors.map(n => (
                      <div key={n.id} className="flex items-center gap-3 rounded bg-muted/30 px-3 py-2">
                        <span className="font-mono text-xs">{n.token}</span>
                        <span className="text-[10px] text-muted-foreground">ID: {n.id}</span>
                        <div className="flex-1">
                          <div className="h-1.5 rounded-full bg-muted">
                            <div className="h-full rounded-full bg-primary" style={{ width: `${Math.min(100, n.score * 100)}%` }} />
                          </div>
                        </div>
                        <span className="font-mono text-[10px] text-muted-foreground">{n.score.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {tab === 'embed' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Token Embedding</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input value={embedQuery} onChange={e => setEmbedQuery(e.target.value)} placeholder="Enter a token..." className="h-8 text-xs" onKeyDown={e => e.key === 'Enter' && void handleEmbed()} />
                <Button onClick={() => void handleEmbed()} disabled={loading} className="shrink-0">Get Embedding</Button>
              </div>
              {embedResult && (
                <div className="space-y-3">
                  <div className="grid grid-cols-3 gap-2">
                    <div className="rounded bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Token</div>
                      <div className="text-xs font-mono font-medium">{embedResult.token}</div>
                    </div>
                    <div className="rounded bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Dimension</div>
                      <div className="text-xs font-mono font-medium">{embedResult.dim}</div>
                    </div>
                    <div className="rounded bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Norm</div>
                      <div className="text-xs font-mono font-medium">{embedResult.norm.toFixed(4)}</div>
                    </div>
                  </div>
                  <div className="flex gap-0.5">
                    {embedResult.top.map(([val, idx], i) => (
                      <div key={i} className="flex flex-col items-center">
                        <div className="w-4 rounded-t" style={{ height: `${Math.abs(val) * 40}px`, backgroundColor: val > 0 ? 'rgb(124 82 196)' : 'rgb(220 80 90)' }} />
                        <span className="text-[8px] text-muted-foreground mt-0.5">{idx}</span>
                      </div>
                    ))}
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    {embedResult.top.map(([val, idx], i) => (
                      <div key={i} className="flex items-center gap-1 text-[10px]">
                        <span className="text-muted-foreground">dim {idx}:</span>
                        <span className="font-mono">{val.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {tab === 'saved' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Saved Trees</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input value={saveName} onChange={e => setSaveName(e.target.value)} placeholder="Tree name..." className="h-8 text-xs" />
                <Button onClick={() => void handleSave()} disabled={!saveName.trim()} className="shrink-0">Save Current</Button>
              </div>
              {savedTrees.length === 0 ? (
                <p className="text-xs text-muted-foreground">No saved trees.</p>
              ) : (
                <div className="space-y-1">
                  {savedTrees.map(t => (
                    <div key={t.name} className="flex items-center gap-3 rounded bg-muted/30 px-3 py-2">
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-medium">{t.name}</div>
                        <div className="text-[10px] text-muted-foreground">{t.vocab_size} vocab, {t.num_merges} merges</div>
                      </div>
                      <Button size="sm" variant="ghost" className="h-6 px-1.5 text-[10px]" onClick={() => void handleLoad(t.name)}>Load</Button>
                      <Button size="sm" variant="ghost" className="h-6 px-1.5 text-[10px] text-destructive" onClick={() => void handleDeleteTree(t.name)}>Delete</Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {tab === 'matrix' && matrixResult && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Embedding Matrix Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <div className="rounded bg-muted/30 p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">Matrix Shape</div>
                  <div className="text-xs font-mono">{matrixResult.matrix ? `${matrixResult.matrix[0]}×${matrixResult.matrix[1]}` : 'N/A'}</div>
                </div>
                <div className="rounded bg-muted/30 p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">Dead Tokens</div>
                  <div className="text-xs font-mono">{matrixResult.dead_tokens}</div>
                </div>
                <div className="rounded bg-muted/30 p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">Live Tokens</div>
                  <div className="text-xs font-mono">{matrixResult.live_tokens}</div>
                </div>
                <div className="rounded bg-muted/30 p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">Norm Range</div>
                  <div className="text-xs font-mono">{matrixResult.norm_min.toFixed(3)}–{matrixResult.norm_max.toFixed(3)}</div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-[10px] text-muted-foreground mb-1">Most Energetic</p>
                  {matrixResult.most_energetic.map(([token, id, norm], i) => (
                    <div key={i} className="flex items-center gap-2 text-[10px]">
                      <span className="font-mono">{token}</span>
                      <span className="text-muted-foreground">ID:{id}</span>
                      <span className="font-mono text-primary">{norm.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground mb-1">Least Energetic</p>
                  {matrixResult.least_energetic.map(([token, id, norm], i) => (
                    <div key={i} className="flex items-center gap-2 text-[10px]">
                      <span className="font-mono">{token}</span>
                      <span className="text-muted-foreground">ID:{id}</span>
                      <span className="font-mono text-muted-foreground">{norm.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {tab === 'compare' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Compare Tokens</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label className="text-xs">Token A</Label>
                  <Input value={compareA} onChange={e => setCompareA(e.target.value)} className="h-8 text-xs font-mono" />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Token B</Label>
                  <Input value={compareB} onChange={e => setCompareB(e.target.value)} className="h-8 text-xs font-mono" />
                </div>
              </div>
              <Button onClick={() => void handleCompare()} disabled={loading || !compareA.trim() || !compareB.trim()} className="w-full">
                {loading ? 'Comparing...' : 'Compare'}
              </Button>
              {compareResult && (
                <div className="space-y-3">
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { label: 'Shared Tokens', value: compareResult.shared_tokens },
                      { label: 'Only A', value: compareResult.only_a_tokens },
                      { label: 'Only B', value: compareResult.only_b_tokens },
                      { label: 'Shared Merges', value: compareResult.shared_merges },
                      { label: 'Only A Merges', value: compareResult.only_a_merges },
                      { label: 'Only B Merges', value: compareResult.only_b_merges },
                    ].map(s => (
                      <div key={s.label} className="rounded bg-muted/30 p-2 text-center">
                        <div className="text-[10px] text-muted-foreground">{s.label}</div>
                        <div className="text-xs font-mono font-medium">{s.value}</div>
                      </div>
                    ))}
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">Shared</p>
                      {compareResult.shared_examples.map(([t, c], i) => (
                        <div key={i} className="text-[10px] font-mono">{t} ({c})</div>
                      ))}
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">Only {compareResult.a.name}</p>
                      {compareResult.only_a_examples.map(([t, c], i) => (
                        <div key={i} className="text-[10px] font-mono">{t} ({c})</div>
                      ))}
                    </div>
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">Only {compareResult.b.name}</p>
                      {compareResult.only_b_examples.map(([t, c], i) => (
                        <div key={i} className="text-[10px] font-mono">{t} ({c})</div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
