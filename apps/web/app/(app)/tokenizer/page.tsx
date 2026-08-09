'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Textarea, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { tokenizerController, type TokenizerStats, type SampleWord } from '@/lib/tokenizer-controller'
import { TokenizerEfficiencyCard } from '@/components/tokenizer/TokenizerEfficiencyCard'
import { useToastStore } from '@/lib/toast-store'

type Tab = 'playground' | 'vocab' | 'samples' | 'train'

export default function TokenizerPage() {
  const [stats, setStats] = useState<TokenizerStats | null>(null)
  const [tab, setTab] = useState<Tab>('playground')
  const [loading, setLoading] = useState(true)

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
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    tokenizerController.getStats().then(s => {
      setStats(s)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const handleTokenize = async () => {
    if (!inputText.trim()) return
    setTokenizing(true)
    try {
      const res = await tokenizerController.tokenize(inputText)
      setTokenResult(res)
    } catch {
      addToast('Tokenization failed', 'error')
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
      addToast('Failed to load vocabulary', 'error')
    }
  }

  const handleLoadSamples = async () => {
    try {
      const res = await tokenizerController.getSamples()
      setSamples(res.samples)
    } catch {
      addToast('Failed to load samples', 'error')
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
      setTrainResult(err instanceof Error ? err.message : 'Training failed')
    } finally {
      setTraining(false)
    }
  }

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Tokenizer" subtitle="BPE tokenizer management" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Tokenizer" subtitle={stats ? `Vocab: ${stats.vocab_size} · Merges: ${stats.total_merges}` : 'BPE tokenizer'} />} />
      <div className="space-y-4">
        {stats && (
          <KpiGrid>
            <StatCard label="Vocab Size" value={String(stats.vocab_size)} />
            <StatCard label="Base Chars" value={String(stats.base_chars)} />
            <StatCard label="Merges" value={String(stats.total_merges)} />
            <StatCard label="Special Tokens" value={String(stats.special_tokens)} />
          </KpiGrid>
        )}

        <TokenizerEfficiencyCard stats={stats} samples={samples} />

        <div className="flex gap-1 border-b border-border/30 pb-0">
          {(['playground', 'vocab', 'samples', 'train'] as Tab[]).map(t => (
            <button
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
      </div>
    </div>
  )
}
