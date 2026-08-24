'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Textarea, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { tokenizerController, type TokenizerStats, type SampleWord } from '@/lib/tokenizer-controller'
import { TokenizerEfficiencyCard } from '@/components/tokenizer/TokenizerEfficiencyCard'
import { TokenTreeQueryCard } from '@/components/tokenizer/TokenTreeQueryCard'
import { TokenTreeTrainCard } from '@/components/tokenizer/TokenTreeTrainCard'
import { TokenTreeCodecCard } from '@/components/tokenizer/TokenTreeCodecCard'
import { TokenTreeMergesCard } from '@/components/tokenizer/TokenTreeMergesCard'
import { TokenTreeVocabCard } from '@/components/tokenizer/TokenTreeVocabCard'
import { TokenTreePersistenceCard } from '@/components/tokenizer/TokenTreePersistenceCard'
import { TokenTreeEmbeddingsCard } from '@/components/tokenizer/TokenTreeEmbeddingsCard'
import { TokenTreeMatrixCard } from '@/components/tokenizer/TokenTreeMatrixCard'
import { TokenTreePathCard } from '@/components/tokenizer/TokenTreePathCard'
import { TokenTreeLineageCard } from '@/components/tokenizer/TokenTreeLineageCard'
import { TokenTreeCompareCard } from '@/components/tokenizer/TokenTreeCompareCard'
import { TokenTreePlaygroundCard } from '@/components/tokenizer/TokenTreePlaygroundCard'
import { useToastStore } from '@/lib/toast-store'

type Tab = 'playground' | 'vocab' | 'samples' | 'train' | 'analyze' | 'decompose' | 'detokenize' | 'pretokenize' | 'merges'

export default function TokenizerPage() {
  const router = useRouter()
  const [stats, setStats] = useState<TokenizerStats | null>(null)
  const [tab, setTab] = useState<Tab>('playground')
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [inputText, setInputText] = useState('')
  const [tokenResult, setTokenResult] = useState<{ tokens: string[]; ids: number[] } | null>(null)
  const [tokenizing, setTokenizing] = useState(false)

  const [vocabEntries, setVocabEntries] = useState<{ id: number; token: string; is_special: boolean }[]>([])
  const [vocabTotal, setVocabTotal] = useState(0)
  const [vocabOffset, setVocabOffset] = useState(0)

  const [samples, setSamples] = useState<SampleWord[]>([])

  const [trainVocab, setTrainVocab] = useState(512)
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState<string | null>(null)
  const [analyzeResult, setAnalyzeResult] = useState<string | null>(null)
  const [decomposeResult, setDecomposeResult] = useState<Record<string, unknown> | null>(null)
  const [detokenizeResult, setDetokenizeResult] = useState<string | null>(null)
  const [mergesResult, setMergesResult] = useState<import('@/lib/tokenizer-controller').MergeEntry[] | null>(null)
  const [mergesQuery, setMergesQuery] = useState('')
  const [pretokenizeResult, setPretokenizeResult] = useState<Record<string, unknown> | null>(null)
  const [treeVersion, setTreeVersion] = useState(0)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    tokenizerController.getStats().then(s => {
      setStats(s)
      setLoading(false)
    }).catch(() => {
      setLoadError('Could not load tokenizer data. Please try again.')
      setLoading(false)
    })
  }, [])

  const handleTokenize = async () => {
    if (!inputText.trim()) return
    setTokenizing(true)
    try {
      const res = await tokenizerController.tokenize(inputText)
      setTokenResult(res)
    } catch {
      addToast('Could not tokenization', 'error')
    } finally {
      setTokenizing(false)
    }
  }

  const handleLoadVocab = async (offset = 0) => {
    try {
      const res = await tokenizerController.getVocab(50, offset)
      setVocabEntries(res.entries)
      setVocabTotal(res.total)
      setVocabOffset(offset)
    } catch {
      addToast('Could not load vocabulary', 'error')
    }
  }

  const handleLoadSamples = async () => {
    try {
      const res = await tokenizerController.getSamples()
      setSamples(res.samples)
    } catch {
      addToast('Could not load samples', 'error')
    }
  }

  const handleTrain = async () => {
    setTraining(true)
    setTrainResult(null)
    try {
      const res = await tokenizerController.train({ vocab_size: trainVocab })
      setTrainResult(`Trained on ${res.corpus_size} lines. Vocab: ${res.stats.vocab_size}`)
      setStats(res.stats)
    } catch (err) {
      setTrainResult(err instanceof Error ? err.message : 'Could not training')
    } finally {
      setTraining(false)
    }
  }


  const handleAnalyze = async () => {
    if (!inputText.trim()) return
    setAnalyzeResult(null)
    try {
      const res = await tokenizerController.analyze([inputText])
      setAnalyzeResult(JSON.stringify(res, null, 2))
    } catch (err) {
      setAnalyzeResult(err instanceof Error ? err.message : 'Could not analyze')
    }
  }

  const handleDecompose = async () => {
    if (!inputText.trim()) return
    setDecomposeResult(null)
    try {
      const res = await tokenizerController.decompose(inputText)
      setDecomposeResult(res)
    } catch (err) {
      setDecomposeResult({ error: err instanceof Error ? err.message : 'Could not decompose' })
    }
  }

  const handleDetokenize = async () => {
    if (!inputText.trim()) return
    setDetokenizeResult(null)
    try {
      const res = await tokenizerController.detokenize(inputText.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)))
      setDetokenizeResult(res.text)
    } catch (err) {
      setDetokenizeResult(err instanceof Error ? err.message : 'Could not detokenize')
    }
  }

const handleGetMerges = async () => {
    try {
      const res = await tokenizerController.getMerges(50)
      setMergesResult(res.merges)
    } catch {
      setMergesResult([])
    }
  }

  const handlePretokenize = async () => {
    if (!inputText.trim()) return
    setPretokenizeResult(null)
    try {
      const res = await tokenizerController.pretokenize(inputText)
      setPretokenizeResult(res)
    } catch (err) {
      setPretokenizeResult({ error: err instanceof Error ? err.message : 'Could not pretokenize' })
    }
  }

  const toolbar = (
    <div className="flex gap-1 border-b border-border/30 pb-0">
      {(['playground', 'vocab', 'samples', 'train', 'analyze', 'decompose', 'detokenize', 'pretokenize', 'merges'] as Tab[]).map(t => (
        <button
          type="button"
          key={t}
          onClick={() => {
            setTab(t)
            if (t === 'vocab') handleLoadVocab()
            if (t === 'samples') handleLoadSamples()
          }}
          className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors ${
            tab === t ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          {t.charAt(0).toUpperCase() + t.slice(1)}
        </button>
      ))}
    </div>
  )

  return (
    <PageContainer
      title="Tokenizer"
      subtitle={stats ? `Vocab: ${stats.vocab_size} · Merges: ${stats.total_merges}` : 'BPE tokenizer'}
      loading={loading}
      error={loadError}
      onRetry={() => window.location.reload()}
      toolbar={toolbar}
    >
      {stats && (
        <KpiGrid>
          <StatCard label="Vocab Size" value={String(stats.vocab_size)} />
          <StatCard label="Base Chars" value={String(stats.base_chars)} />
          <StatCard label="Merges" value={String(stats.total_merges)} />
          <StatCard label="Special Tokens" value={String(stats.special_tokens)} />
        </KpiGrid>
      )}

      {!loading && !loadError && !stats && (
        <div className="text-center py-8 text-sm text-muted-foreground">
          No tokenizer data available. Train a tokenizer or try again later.
          <div className="mt-2">
            <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => router.push('/training')}>
              Go to Training
            </Button>
          </div>
        </div>
      )}

      <TokenizerEfficiencyCard stats={stats} samples={samples} />

      {tab === 'playground' && (
        <Card>
          <CardContent className="pt-4 space-y-3">
            <Textarea
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              placeholder="Enter text to tokenize..."
              rows={3}
            />
            <Button size="sm" onClick={handleTokenize} disabled={tokenizing || !inputText.trim()}>
              {tokenizing ? 'Tokenizing...' : 'Tokenize'}
            </Button>
            {tokenResult && (
              <div className="space-y-2">
                <div>
                  <div className="text-xs text-muted-foreground mb-1">Tokens ({tokenResult.tokens.length})</div>
                  <div className="flex flex-wrap gap-1">
                    {tokenResult.tokens.map((t, i) => (
                      <span key={i} className="text-xs font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground mb-1">IDs</div>
                  <div className="text-xs font-mono text-muted-foreground">[{tokenResult.ids.join(', ')}]</div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'vocab' && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Vocabulary ({vocabTotal})</CardTitle>
            <Button size="sm" variant="ghost" onClick={() => handleLoadVocab(vocabOffset)}>
              <IconRefresh className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-1 max-h-96 overflow-y-auto font-mono text-xs">
              {vocabEntries.map(e => (
                <div key={e.id} className="flex items-center gap-3 py-0.5 border-b border-border/20">
                  <span className="w-12 text-right text-muted-foreground">{e.id}</span>
                  <span className={e.is_special ? 'text-primary font-medium' : ''}>{e.token}</span>
                </div>
              ))}
            </div>
            <div className="flex gap-2 mt-3">
              <Button size="sm" variant="outline" onClick={() => handleLoadVocab(Math.max(0, vocabOffset - 50))} disabled={vocabOffset === 0}>
                Prev
              </Button>
              <Button size="sm" variant="outline" onClick={() => handleLoadVocab(vocabOffset + 50)} disabled={vocabOffset + 50 >= vocabTotal}>
                Next
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {tab === 'samples' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Tokenization Samples</CardTitle>
          </CardHeader>
          <CardContent>
            {samples.length === 0 ? (
              <Button size="sm" onClick={handleLoadSamples}>Load Samples</Button>
            ) : (
              <div className="space-y-2">
                {samples.map(s => (
                  <div key={s.word} className="flex items-center gap-3 text-xs py-1 border-b border-border/20">
                    <span className="font-medium w-24 truncate">{s.word}</span>
                    <div className="flex gap-1">
                      {s.tokens.map((t, i) => (
                        <span key={i} className="font-mono bg-muted/50 px-1 py-0.5 rounded">{t}</span>
                      ))}
                    </div>
                    <span className="text-muted-foreground ml-auto">{s.count} tokens</span>
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
            <CardTitle className="text-base">Train Tokenizer</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {trainResult && (
              <div className="rounded-md bg-primary/10 border border-primary/20 px-3 py-2 text-sm text-primary">
                {trainResult}
              </div>
            )}
            <div className="flex items-center gap-2">
              <label className="text-xs text-muted-foreground">Vocab size:</label>
              <Input
                type="number"
                aria-label="Vocab size"
                value={trainVocab}
                onChange={e => setTrainVocab(parseInt(e.target.value) || 512)}
                className="w-24"
                min={32}
                max={100000}
              />
              <Button size="sm" onClick={handleTrain} disabled={training}>
                {training ? 'Training...' : 'Train on Shakespeare'}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Downloads Shakespeare dataset and trains a BPE tokenizer. Takes a few seconds.
            </p>
          </CardContent>
        </Card>
      )}
      {tab === 'analyze' && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">Enter text to analyze token structure, entropy, and compression.</p>
          <Button size="sm" onClick={() => void handleAnalyze()}>Analyze</Button>
          {analyzeResult && (
            <pre className="rounded bg-muted p-3 text-xs overflow-auto max-h-80 whitespace-pre-wrap">{analyzeResult}</pre>
          )}
        </div>
      )}
      {tab === 'decompose' && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">Enter text to decompose into character-level representation.</p>
          <Button size="sm" onClick={() => void handleDecompose()}>Decompose</Button>
          {decomposeResult && (
            <pre className="rounded bg-muted p-3 text-xs overflow-auto max-h-80 whitespace-pre-wrap">{JSON.stringify(decomposeResult, null, 2)}</pre>
          )}
        </div>
      )}
      {tab === 'detokenize' && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">Enter comma-separated token IDs to convert back to text.</p>
          <Button size="sm" onClick={() => void handleDetokenize()}>Detokenize</Button>
          {detokenizeResult && (
            <pre className="rounded bg-muted p-3 text-xs overflow-auto max-h-80 whitespace-pre-wrap">{detokenizeResult}</pre>
          )}
        </div>
      )}
      
      {tab === 'merges' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">BPE Merges</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-2">
              <Input value={mergesQuery} onChange={e => setMergesQuery(e.target.value)} placeholder="Filter merges (optional)" className="h-8 text-xs flex-1" />
              <Button size="sm" onClick={() => void handleGetMerges()}>Load Merges</Button>
            </div>
            {mergesResult && (
              <div className="rounded bg-muted p-3 text-xs max-h-64 overflow-auto font-mono">
                {mergesResult.length === 0 ? <p className="text-muted-foreground">No merges found</p> : mergesResult.map((m, i) => <div key={i}>{m.index}: {m.left} + {m.right} → {m.token}</div>)}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'pretokenize' && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">Enter text to see how it is pre-tokenized into words/fragments.</p>
          <Button size="sm" onClick={() => void handlePretokenize()}>Pretokenize</Button>
          {pretokenizeResult && (
            <pre className="rounded bg-muted p-3 text-xs overflow-auto max-h-80 whitespace-pre-wrap">{JSON.stringify(pretokenizeResult, null, 2)}</pre>
          )}
        </div>
      )}


      <TokenTreeQueryCard key={treeVersion} />
      <TokenTreeMergesCard refreshKey={treeVersion} />
      <TokenTreeVocabCard refreshKey={treeVersion} />
      <TokenTreeTrainCard onTrained={() => setTreeVersion(v => v + 1)} />
      <TokenTreePersistenceCard refreshKey={treeVersion} onLoaded={() => setTreeVersion(v => v + 1)} />
      <TokenTreeEmbeddingsCard />
      <TokenTreeMatrixCard />
      <TokenTreeCodecCard />
      <TokenTreePathCard />
      <TokenTreeLineageCard />
      <TokenTreeCompareCard key={treeVersion} />
      <TokenTreePlaygroundCard />
    </PageContainer>
  )
}
