'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Textarea, Input, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { tokenTreeController, type TrainTreeResult } from '@/lib/token-tree-controller'
import { useToastStore } from '@/lib/toast-store'

interface TokenTreeTrainCardProps {
  onTrained?: () => void
}

export function TokenTreeTrainCard({ onTrained }: TokenTreeTrainCardProps) {
  const [corpus, setCorpus] = useState('')
  const [vocabSize, setVocabSize] = useState(512)
  const [embedDim, setEmbedDim] = useState(16)
  const [minFreq, setMinFreq] = useState(2)
  const [training, setTraining] = useState(false)
  const [result, setResult] = useState<TrainTreeResult | null>(null)
  const [failed, setFailed] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const handleTrain = async () => {
    setTraining(true)
    setFailed(false)
    const texts = corpus
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
    try {
      const res = await tokenTreeController.train(
        texts.length > 0
          ? { texts, vocab_size: vocabSize, embed_dim: embedDim, min_frequency: minFreq }
          : { vocab_size: vocabSize, embed_dim: embedDim, min_frequency: minFreq },
      )
      setResult(res)
      onTrained?.()
    } catch (err) {
      setFailed(true)
      addToast(err instanceof Error ? err.message : 'Failed to train token tree', 'error')
    } finally {
      setTraining(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Train Token Tree</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {result && !training && (
          <div className="rounded-md bg-success/10 border border-success/20 px-3 py-2 text-sm text-success">
            Trained: vocab {result.vocab_size} · embed dim {result.embed_dim} · compression{' '}
            {result.embedding_compression_ratio}x
          </div>
        )}
        {failed && !training && (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-sm text-destructive">
            Training failed. Check the server logs and try again.
          </div>
        )}
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Corpus
          </label>
          <Textarea
            value={corpus}
            onChange={e => setCorpus(e.target.value)}
            rows={4}
            placeholder="Paste a corpus — one document per line. Leave empty to use the built-in default corpus."
            aria-label="Token tree training corpus"
          />
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <label className="block text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Vocab size
            </label>
            <Input
              type="number"
              value={vocabSize}
              onChange={e => setVocabSize(parseInt(e.target.value) || 512)}
              className="w-24"
              min={32}
              max={100000}
              aria-label="Token tree vocab size"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Embed dim
            </label>
            <Input
              type="number"
              value={embedDim}
              onChange={e => setEmbedDim(parseInt(e.target.value) || 0)}
              className="w-24"
              min={0}
              max={4096}
              aria-label="Token tree embed dim"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Min frequency
            </label>
            <Input
              type="number"
              value={minFreq}
              onChange={e => setMinFreq(parseInt(e.target.value) || 1)}
              className="w-24"
              min={1}
              max={10000}
              aria-label="Token tree min frequency"
            />
          </div>
          <Button size="sm" onClick={handleTrain} disabled={training}>
            {training ? 'Training...' : (
              <>
                <IconRefresh className="h-4 w-4 mr-1" />
                Train token tree
              </>
            )}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Learns BPE merges and co-occurrence embeddings, stored as compressed pugqeep points. The result powers the
          semantic query card above.
        </p>
      </CardContent>
    </Card>
  )
}
