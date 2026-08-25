'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Skeleton,
  Chip,
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from '@sloughgpt/strui'
import { tokenTreeController, type CompareResult, type SavedTree } from '@/lib/token-tree-controller'
import { useToastStore } from '@/lib/toast-store'

interface TokenTreeCompareCardProps {
  refreshKey?: number
}

function ExampleList({ title, examples, accent }: { title: string; examples: [string, number][]; accent: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground mb-1">{title}</div>
      {examples.length === 0 ? (
        <div className="text-xs text-muted-foreground">none</div>
      ) : (
        <div className="flex flex-wrap gap-1">
          {examples.map(([token, freq]) => (
            <span key={token} className={`text-xs font-mono px-1.5 py-0.5 rounded ${accent}`}>
              {token} <span className="opacity-70">×{freq}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export function TokenTreeCompareCard({ refreshKey = 0 }: TokenTreeCompareCardProps) {
  const [trees, setTrees] = useState<SavedTree[]>([])
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  const [comparing, setComparing] = useState(false)
  const [loadFailed, setLoadFailed] = useState(false)
  const [result, setResult] = useState<CompareResult | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const load = useCallback(async () => {
    try {
      const saved = await tokenTreeController.listSaved()
      setTrees(saved)
      setLoadFailed(false)
    } catch {
      setLoadFailed(true)
    }
  }, [])

  useEffect(() => {
    load()
  }, [refreshKey, load])

  const handleCompare = async () => {
    if (!a || !b || a === b) return
    setComparing(true)
    setResult(null)
    try {
      const res = await tokenTreeController.compare(a, b, 10)
      setResult(res)
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Could not compare token trees', 'error')
    } finally {
      setComparing(false)
    }
  }

  const names = trees.map(t => t.name)
  const hasTwo = trees.length >= 2

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Compare Token Trees</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {loadFailed ? (
          <div className="text-center py-4 text-sm text-muted-foreground">
            Could not load saved trees. <button type="button" onClick={load} className="text-primary underline">Retry</button>
          </div>
        ) : trees.length === 0 ? (
          <div className="text-center py-6 text-sm text-muted-foreground">
            No saved trees yet — save at least two trees to compare vocabularies and merge rules.
          </div>
        ) : (
          <>
            <div className="flex items-end gap-2">
              <div className="flex-1 space-y-1">
                <label className="block text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Tree A
                </label>
                <Select value={a} onValueChange={setA}>
                  <SelectTrigger placeholder="Choose first tree">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {names.map(n => (
                      <SelectItem key={n} value={n}>
                        {n}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex-1 space-y-1">
                <label className="block text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Tree B
                </label>
                <Select value={b} onValueChange={setB}>
                  <SelectTrigger placeholder="Choose second tree">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {names.map(n => (
                      <SelectItem key={n} value={n}>
                        {n}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button size="sm" onClick={handleCompare} disabled={!hasTwo || !a || !b || a === b || comparing}>
                {comparing ? 'Comparing...' : 'Compare'}
              </Button>
            </div>
            {a && b && a === b && (
              <p className="text-xs text-destructive">Choose two different trees to compare.</p>
            )}
            {!hasTwo && (
              <p className="text-xs text-muted-foreground">
                Comparing needs at least two saved trees. Save the current tree to create a second one.
              </p>
            )}

            {result ? (
              <div className="space-y-3 pt-1">
                <div className="flex flex-wrap gap-2">
                  <Chip label={`${result.a.name} · vocab ${result.a.stats.vocab_size}`} />
                  <Chip label={`${result.b.name} · vocab ${result.b.stats.vocab_size}`} />
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="rounded-md bg-muted/50 px-2 py-1.5">
                    <div className="text-lg font-medium">{result.shared_tokens}</div>
                    <div className="text-[10px] text-muted-foreground uppercase tracking-wider">shared tokens</div>
                  </div>
                  <div className="rounded-md bg-muted/50 px-2 py-1.5">
                    <div className="text-lg font-medium">{result.only_a_tokens}</div>
                    <div className="text-[10px] text-muted-foreground uppercase tracking-wider">only in A</div>
                  </div>
                  <div className="rounded-md bg-muted/50 px-2 py-1.5">
                    <div className="text-lg font-medium">{result.only_b_tokens}</div>
                    <div className="text-[10px] text-muted-foreground uppercase tracking-wider">only in B</div>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 text-center">
                  <Chip label={`${result.shared_merges} shared merges`} />
                  <Chip label={`${result.only_a_merges} merges in A`} />
                  <Chip label={`${result.only_b_merges} merges in B`} />
                </div>
                <ExampleList title={`Shared token examples (${result.shared_examples.length})`} examples={result.shared_examples} accent="bg-primary/10 text-primary" />
                <ExampleList title={`Only in ${result.a.name} (${result.only_a_examples.length})`} examples={result.only_a_examples} accent="bg-accent/10 text-accent" />
                <ExampleList title={`Only in ${result.b.name} (${result.only_b_examples.length})`} examples={result.only_b_examples} accent="bg-success/10 text-success" />
              </div>
            ) : comparing ? (
              <div className="space-y-2">
                <Skeleton className="h-16 w-full rounded" />
                <Skeleton className="h-8 w-full rounded" />
              </div>
            ) : null}
          </>
        )}
        <p className="text-xs text-muted-foreground">
          Diffs the saved vocabularies and merge rules without changing the current tree. Examples show the highest-frequency tokens of each group.
        </p>
      </CardContent>
    </Card>
  )
}
