'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Chip } from '@sloughgpt/strui'
import { IconSearch } from '@sloughgpt/strui'
import { tokenTreeController, type PathResult } from '@/lib/token-tree-controller'
import { useToastStore } from '@/lib/toast-store'

const displayToken = (token: string) => token.replace('</w>', '').trim() || token

export function TokenTreePathCard() {
  const [text, setText] = useState('the quick brown fox')
  const [result, setResult] = useState<PathResult | null>(null)
  const [tracing, setTracing] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const handleTrace = async () => {
    const term = text.trim()
    if (!term) return
    setTracing(true)
    setResult(null)
    try {
      setResult(await tokenTreeController.path(term))
    } catch {
      addToast('Failed to trace the token path', 'error')
    } finally {
      setTracing(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Token Path Explorer</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-end gap-2">
          <div className="flex-1 space-y-1">
            <label className="block text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Text
            </label>
            <Input
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') handleTrace()
              }}
              placeholder="Text to walk through the trie"
              aria-label="Text to trace"
            />
          </div>
          <Button size="sm" onClick={handleTrace} disabled={!text.trim() || tracing}>
            {tracing ? (
              'Tracing...'
            ) : (
              <>
                <IconSearch className="h-4 w-4 mr-1" />
                Trace
              </>
            )}
          </Button>
        </div>

        {result && (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-1">
              {result.steps.map((step, i) => (
                <Chip key={i} label={displayToken(step.token)} />
              ))}
            </div>

            <div className="rounded-md border border-border/60 divide-y divide-border/30">
              {result.steps.map((step, i) => (
                <div key={i} className="flex items-center justify-between gap-3 px-3 py-1.5 text-xs">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-muted-foreground tabular-nums w-6 shrink-0">#{i + 1}</span>
                    <span className="font-mono truncate text-muted-foreground">{step.remaining}</span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="font-mono text-primary">→ {displayToken(step.token)}</span>
                    <span className="text-muted-foreground tabular-nums">id {step.id}</span>
                    <span className="text-muted-foreground tabular-nums">+{step.consumed}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="text-xs font-mono break-all text-muted-foreground">
              [{result.ids.join(', ')}]
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          Shows the greedy longest-prefix walk: each step consumes the longest token that matches the start of the
          remaining suffix. The token list is identical to the Codec card&apos;s encode — this card reveals how it gets
          there.
        </p>
      </CardContent>
    </Card>
  )
}
