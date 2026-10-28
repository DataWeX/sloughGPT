'use client'

import { useRouter } from '@/vite/next-compat/navigation'
import { useState, useEffect } from 'react'
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Input,
  Textarea,
  StatCard,
  KpiGrid,
  cn,
  FoldSection,
} from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import {
  tokenizerController,
  type TokenizerStats,
  type SampleWord,
} from '@/lib/tokenizer-controller'
import { TokenizerEfficiencyCard } from '@/components/tokenizer/TokenizerEfficiencyCard'
import { clampNumber } from '@/lib/sanitize'
import { TokenTreeQueryCard } from '@/components/tokenizer/TokenTreeQueryCard'
import { TokenTreeMergesCard } from '@/components/tokenizer/TokenTreeMergesCard'
import { TokenTreeVocabCard } from '@/components/tokenizer/TokenTreeVocabCard'
import { TokenTreeTrainCard } from '@/components/tokenizer/TokenTreeTrainCard'
import { TokenTreePersistenceCard } from '@/components/tokenizer/TokenTreePersistenceCard'
import { useToastStore } from '@/lib/toast-store'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

type Tab = 'play' | 'vocab' | 'train'

export default function TokenizerPage() {
  const router = useRouter()
  const [stats, setStats] = useState<TokenizerStats | null>(null)
  const [tab, setTab] = useState<Tab>('play')
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [inputText, setInputText] = useState('')
  const [tokenResult, setTokenResult] = useState<{ tokens: string[]; ids: number[] } | null>(null)
  const [tokenizing, setTokenizing] = useState(false)
  const [vocabEntries, setVocabEntries] = useState<
    { id: number; token: string; is_special: boolean }[]
  >([])
  const [vocabTotal, setVocabTotal] = useState(0)
  const [vocabOffset, setVocabOffset] = useState(0)
  const [samples, setSamples] = useState<SampleWord[]>([])
  const [trainVocab, setTrainVocab] = useState(512)
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState<string | null>(null)
  const [treeVersion, setTreeVersion] = useState(0)
  const addToast = useToastStore((s) => s.addToast)
  useRefreshShortcut(() => router.refresh())

  useEffect(() => {
    tokenizerController
      .getStats()
      .then((s) => {
        setStats(s)
        setLoading(false)
      })
      .catch(() => {
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
      addToast('Could not tokenize', 'error')
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
      setTrainResult(err instanceof Error ? err.message : 'Training failed')
    } finally {
      setTraining(false)
    }
  }

  const onTabChange = (t: Tab) => {
    setTab(t)
    if (t === 'vocab') {
      handleLoadVocab()
      handleLoadSamples()
    }
  }

  const toolbar = (
    <div className="flex gap-1 border-b border-border/30 pb-0">
      {(['play', 'vocab', 'train'] as Tab[]).map((t) => (
        <button
          key={t}
          onClick={() => onTabChange(t)}
          className={cn(
            'px-3 py-1.5 text-xs font-medium rounded-t transition-colors',
            tab === t
              ? 'bg-primary/10 text-primary border-b-2 border-primary'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          {t === 'play' ? 'Play' : t === 'vocab' ? 'Vocabulary' : 'Train'}
        </button>
      ))}
    </div>
  )

  return (
    <PageContainer
      title="Tokenizer"
      subtitle={
        stats
          ? `Vocab: ${stats.vocab_size} · Merges: ${stats.total_merges}`
          : 'BPE tokenizer — try it, peek under hood, train your own'
      }
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
        <div className="text-center py-8 text-xs text-muted-foreground">
          No tokenizer data. Train below or go to Training.
          <div className="mt-2">
            <Button size="sm" variant="outline" onClick={() => router.push('/training')}>
              Go to Training
            </Button>
          </div>
        </div>
      )}

      {tab === 'play' && (
        <div className="space-y-4">
          <TokenizerEfficiencyCard stats={stats} samples={samples} />
          <Card>
            <CardContent className="px-3 pb-3 pt-4 space-y-3">
              <Textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to tokenize..."
                rows={3}
              />
              <Button size="sm" onClick={handleTokenize} disabled={tokenizing || !inputText.trim()}>
                {tokenizing ? 'Tokenizing...' : 'Tokenize'}
              </Button>
              {tokenResult && (
                <div className="space-y-2">
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">
                      Tokens ({tokenResult.tokens.length})
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {tokenResult.tokens.map((t, i) => (
                        <span
                          key={i}
                          className="text-xs font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">IDs</div>
                    <div className="text-xs font-mono text-muted-foreground">
                      [{tokenResult.ids.join(', ')}]
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
          <FoldSection heading="Similar tokens">
            <TokenTreeQueryCard key={treeVersion} />
          </FoldSection>
        </div>
      )}

      {tab === 'vocab' && (
        <div className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-xs font-medium">Vocabulary ({vocabTotal})</CardTitle>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 text-xs"
                onClick={() => handleLoadVocab(vocabOffset)}
              >
                <IconRefresh className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5">
              <div className="space-y-1 max-h-72 overflow-y-auto font-mono text-xs">
                {vocabEntries.map((e) => (
                  <div key={e.id} className="flex gap-1.5 py-0.5 border-b border-border/20">
                    <span className="w-12 text-right text-muted-foreground">{e.id}</span>
                    <span className={e.is_special ? 'text-primary font-medium' : ''}>
                      {e.token}
                    </span>
                  </div>
                ))}
              </div>
              <div className="flex gap-2 mt-3">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleLoadVocab(Math.max(0, vocabOffset - 50))}
                  disabled={vocabOffset === 0}
                >
                  Prev
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleLoadVocab(vocabOffset + 50)}
                  disabled={vocabOffset + 50 >= vocabTotal}
                >
                  Next
                </Button>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-xs font-medium">Samples</CardTitle>
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5">
              {samples.length === 0 ? (
                <Button size="sm" onClick={handleLoadSamples}>
                  Load Samples
                </Button>
              ) : (
                <div className="space-y-1">
                  {samples.map((s) => (
                    <div
                      key={s.word}
                      className="flex gap-1.5 text-xs py-1 border-b border-border/20"
                    >
                      <span className="font-medium w-24 truncate">{s.word}</span>
                      <div className="flex gap-1">
                        {s.tokens.map((t, i) => (
                          <span key={i} className="font-mono bg-muted/20 rounded px-1 py-0.5">
                            {t}
                          </span>
                        ))}
                      </div>
                      <span className="text-muted-foreground ml-auto">{s.count} tokens</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
          <FoldSection heading="Merge rules">
            <TokenTreeMergesCard refreshKey={treeVersion} />
          </FoldSection>
          <FoldSection heading="Tree vocabulary">
            <TokenTreeVocabCard refreshKey={treeVersion} />
          </FoldSection>
        </div>
      )}

      {tab === 'train' && (
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2 pt-2.5 px-2.5">
              <CardTitle className="text-xs font-medium">Train Tokenizer</CardTitle>
            </CardHeader>
            <CardContent className="px-2.5 pb-2.5 space-y-3">
              {trainResult && (
                <div className="rounded bg-primary/10 border border-primary/20 px-3 py-2 text-xs text-primary">
                  {trainResult}
                </div>
              )}
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted-foreground">Vocab size:</label>
                <Input
                  type="number"
                  aria-label="Vocab size"
                  value={trainVocab}
                  onChange={(e) =>
                    setTrainVocab(clampNumber(parseInt(e.target.value) || 512, 2, 50_000))
                  }
                  className="w-24"
                  min={32}
                  max={100000}
                />
                <Button size="sm" onClick={handleTrain} disabled={training}>
                  {training ? 'Training...' : 'Train on Shakespeare'}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Downloads Shakespeare dataset and trains a BPE tokenizer.
              </p>
            </CardContent>
          </Card>
          <FoldSection heading="Train from custom corpus">
            <TokenTreeTrainCard onTrained={() => setTreeVersion((v) => v + 1)} />
          </FoldSection>
          <FoldSection heading="Save & load trees">
            <TokenTreePersistenceCard
              refreshKey={treeVersion}
              onLoaded={() => setTreeVersion((v) => v + 1)}
            />
          </FoldSection>
        </div>
      )}
    </PageContainer>
  )
}
