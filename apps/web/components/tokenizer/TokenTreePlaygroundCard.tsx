'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Textarea, Button, Chip } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { tokenizerController, type TokenizeResult } from '@/lib/tokenizer-controller'
import { tokenTreeController, type EncodeResult } from '@/lib/token-tree-controller'
import { useToastStore } from '@/lib/toast-store'

const displayToken = (token: string) => token.replace('</w>', '').trim() || token

interface PlaygroundResult {
  base: TokenizeResult | null
  tree: EncodeResult | null
}

export function TokenTreePlaygroundCard() {
  const [text, setText] = useState('the quick brown fox jumps over the lazy dog')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<PlaygroundResult | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const handleCompare = async () => {
    const term = text.trim()
    if (!term) return
    setRunning(true)
    setResult(null)
    const [baseRes, treeRes] = await Promise.allSettled([
      tokenizerController.tokenize(term),
      tokenTreeController.encode(term),
    ])
    setResult({
      base: baseRes.status === 'fulfilled' ? baseRes.value : null,
      tree: treeRes.status === 'fulfilled' ? treeRes.value : null,
    })
    if (baseRes.status === 'rejected' && treeRes.status === 'rejected') {
      addToast('Both tokenizers failed. Is the tokenizer trained?', 'error')
    }
    setRunning(false)
  }

  const baseCount = result?.base?.ids.length ?? null
  const treeCount = result?.tree?.ids.length ?? null
  const ratio =
    baseCount !== null && treeCount !== null && treeCount > 0 ? baseCount / treeCount : null

  const verdict =
    ratio === null
      ? null
      : ratio > 1.02
        ? `Token tree is ${Math.round((ratio - 1) * 100)}% more compact`
        : ratio < 0.98
          ? `Base tokenizer is ${Math.round((1 / ratio - 1) * 100)}% more compact`
          : 'Both tokenizers produce the same token count'

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Token Tree Playground</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Text
          </label>
          <Textarea
            value={text}
            onChange={e => setText(e.target.value)}
            rows={2}
            placeholder="Compare how both tokenizers split the same text"
            aria-label="Text to compare"
          />
          <div>
            <Button size="sm" onClick={handleCompare} disabled={running || !text.trim()}>
              {running ? 'Comparing...' : (
                <>
                  <IconRefresh className="h-4 w-4 mr-1" />
                  Compare tokenizers
                </>
              )}
            </Button>
          </div>
        </div>

        {verdict && (
          <div className="rounded-md bg-primary/10 border border-primary/20 px-3 py-2 text-sm text-primary">
            {verdict}
          </div>
        )}

        {result && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-md bg-muted/50 px-3 py-2 space-y-2">
              <div className="text-xs text-muted-foreground">
                Base BPE — {result.base ? `${result.base.ids.length} tokens` : 'unavailable'}
              </div>
              {result.base ? (
                <>
                  <div className="flex flex-wrap gap-1">
                    {result.base.tokens.map((token, i) => (
                      <Chip key={i} label={displayToken(token)} />
                    ))}
                  </div>
                  <div className="text-xs font-mono break-all text-muted-foreground">
                    [{result.base.ids.join(', ')}]
                  </div>
                </>
              ) : (
                <div className="text-xs text-muted-foreground">Tokenizer not trained.</div>
              )}
            </div>
            <div className="rounded-md bg-muted/50 px-3 py-2 space-y-2">
              <div className="text-xs text-muted-foreground">
                Token tree — {result.tree ? `${result.tree.ids.length} tokens` : 'unavailable'}
              </div>
              {result.tree ? (
                <>
                  <div className="flex flex-wrap gap-1">
                    {result.tree.tokens.map((token, i) => (
                      <Chip key={i} label={displayToken(token)} />
                    ))}
                  </div>
                  <div className="text-xs font-mono break-all text-muted-foreground">
                    [{result.tree.ids.join(', ')}]
                  </div>
                </>
              ) : (
                <div className="text-xs text-muted-foreground">Token tree not trained.</div>
              )}
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          Splits the same text with the base BPE tokenizer and the trained token tree side by side.
          The token count difference shows how much the merge tree compresses the corpus.
        </p>
      </CardContent>
    </Card>
  )
}
