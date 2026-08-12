'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Chip, Skeleton } from '@sloughgpt/strui'
import { IconActivity } from '@sloughgpt/strui'
import { tokenTreeController, type LineageResult } from '@/lib/token-tree-controller'
import { useToastStore } from '@/lib/toast-store'

const displayToken = (token: string) => token.replace('</w>', '').trim() || token

export function TokenTreeLineageCard() {
  const [input, setInput] = useState('')
  const [result, setResult] = useState<LineageResult | null>(null)
  const [loading, setLoading] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const handleLookup = async () => {
    const term = input.trim()
    if (!term) return
    setLoading(true)
    setResult(null)
    try {
      setResult(await tokenTreeController.lineage(term))
    } catch {
      addToast(`Token not in vocabulary: ${term}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Merge Lineage Explorer</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <Input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') handleLookup()
            }}
            placeholder="Token or id, e.g. quick or 12"
            className="max-w-xs"
            aria-label="Token to inspect"
          />
          <Button size="sm" onClick={handleLookup} disabled={loading || !input.trim()}>
            {loading ? 'Loading...' : (
              <>
                <IconActivity className="h-4 w-4 mr-1" />
                Show lineage
              </>
            )}
          </Button>
        </div>

        {loading ? (
          <Skeleton className="h-16 w-full rounded" />
        ) : result ? (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              Merge lineage of{' '}
              <span className="font-mono text-primary">"{displayToken(result.token)}"</span>
              <span className="text-muted-foreground/70"> — {result.leaves.length} character leaves</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {result.leaves.map((leaf, i) => (
                <Chip key={i} label={leaf} />
              ))}
            </div>
            <pre className="rounded-md bg-muted/50 px-3 py-2 text-xs font-mono whitespace-pre-wrap break-words text-muted-foreground">
              {result.tree}
            </pre>
          </div>
        ) : null}

        <p className="text-xs text-muted-foreground">
          Every BPE token is a merge of two shorter pieces. This card walks a token's merge history down to its
          character leaves and renders the merge tree exactly as the core builds it.
        </p>
      </CardContent>
    </Card>
  )
}
