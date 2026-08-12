'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Chip } from '@sloughgpt/strui'
import { tokenTreeController, type EmbeddingInfo } from '@/lib/token-tree-controller'

const TOP_K = 8

export function TokenTreeEmbeddingsCard() {
  const [token, setToken] = useState('quick')
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)
  const [result, setResult] = useState<EmbeddingInfo | null>(null)

  const handleInspect = async () => {
    const term = token.trim()
    if (!term) return
    setLoading(true)
    setFailed(false)
    setResult(null)
    try {
      setResult(await tokenTreeController.getEmbedding(term, TOP_K))
    } catch {
      setFailed(true)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Token Embedding Explorer</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-end gap-2">
          <div className="flex-1 space-y-1">
            <label className="block text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Token
            </label>
            <Input
              value={token}
              onChange={e => setToken(e.target.value)}
              placeholder="Token literal or id (e.g. quick or 12)"
              aria-label="Token to inspect"
              onKeyDown={e => {
                if (e.key === 'Enter') handleInspect()
              }}
            />
          </div>
          <Button size="sm" onClick={handleInspect} disabled={!token.trim() || loading}>
            {loading ? 'Inspecting...' : 'Inspect'}
          </Button>
        </div>

        {failed && !result && (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-sm text-destructive">
            Token not in the vocabulary, or embeddings are disabled. Try a token shown in the Vocabulary card.
          </div>
        )}

        {result && !failed && (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              <Chip label={`"${result.token}" · id ${result.id}`} />
              <Chip label={`Dim ${result.dim}`} />
              <Chip label={`L2 norm ${result.norm.toFixed(4)}`} />
              <Chip label={`${result.embedding_points} points`} />
              <Chip label={`${result.compression_ratio}x compressed`} />
            </div>

            <div className="space-y-1">
              <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Largest dimensions by magnitude
              </div>
              <div className="divide-y divide-border/30">
                {result.top.map(([dim, value]) => (
                  <div key={dim} className="flex items-center justify-between py-1 text-xs font-mono">
                    <span className="text-muted-foreground">dim {dim}</span>
                    <span className={value >= 0 ? 'text-primary' : 'text-destructive'}>
                      {value >= 0 ? '+' : ''}
                      {value.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <p className="text-xs text-muted-foreground">
              Each token's embedding is generated from a compressed pugqeep point, not stored as a full vector — the
              same substrate that powers the semantic query card. The vector below is what feeds cosine similarity.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
