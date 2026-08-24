'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Skeleton, Chip } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { tokenTreeController, type SavedTree, type TokenTreeStats } from '@/lib/token-tree-controller'
import { useToastStore } from '@/lib/toast-store'

interface TokenTreePersistenceCardProps {
  refreshKey?: number
  onLoaded?: () => void
}

const formatDate = (savedAt: number | null) => {
  if (!savedAt) return 'unknown time'
  return new Date(savedAt * 1000).toLocaleString()
}

export function TokenTreePersistenceCard({ refreshKey = 0, onLoaded }: TokenTreePersistenceCardProps) {
  const [stats, setStats] = useState<TokenTreeStats | null>(null)
  const [trees, setTrees] = useState<SavedTree[]>([])
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)
  const [busyName, setBusyName] = useState<string | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const load = useCallback(async () => {
    try {
      const [s, saved] = await Promise.all([
        tokenTreeController.getStats(),
        tokenTreeController.listSaved(),
      ])
      setStats(s)
      setTrees(saved)
      setLoadFailed(false)
    } catch {
      setLoadFailed(true)
    }
  }, [])

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey])

  const handleSave = async () => {
    const trimmed = name.trim()
    if (!trimmed) return
    setSaving(true)
    try {
      await tokenTreeController.saveTree(trimmed)
      setName('')
      addToast(`Saved token tree "${trimmed}"`, 'success')
      await load()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Could not save token tree', 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleLoad = async (tree: SavedTree) => {
    setBusyName(tree.name)
    try {
      await tokenTreeController.loadTree(tree.name)
      addToast(`Loaded token tree "${tree.name}"`, 'success')
      onLoaded?.()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Could not load token tree', 'error')
    } finally {
      setBusyName(null)
    }
  }

  const handleDelete = async (tree: SavedTree) => {
    setBusyName(tree.name)
    try {
      await tokenTreeController.deleteSavedTree(tree.name)
      addToast(`Deleted token tree "${tree.name}"`, 'success')
      await load()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Could not delete token tree', 'error')
    } finally {
      setBusyName(null)
    }
  }

  const trained = stats?.trained ?? false

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Saved Token Trees</CardTitle>
        <Button size="sm" variant="ghost" onClick={load} disabled={loadFailed} aria-label="Refresh saved trees">
          <IconRefresh className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {loadFailed ? (
          <div className="text-center py-4 text-sm text-muted-foreground">
            Could not load saved trees. <button onClick={load} className="text-primary underline">Retry</button>
          </div>
        ) : stats === null ? (
          <div className="space-y-2">
            <Skeleton className="h-9 w-full rounded" />
            <Skeleton className="h-6 w-full rounded" />
          </div>
        ) : (
          <>
            <div className="flex items-end gap-2">
              <div className="flex-1 space-y-1">
                <label className="block text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Save current tree
                </label>
                <Input
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="tree-name (letters, digits, dots, dashes)"
                  aria-label="Saved tree name"
                  disabled={!trained}
                  onKeyDown={e => {
                    if (e.key === 'Enter') handleSave()
                  }}
                />
              </div>
              <Button size="sm" onClick={handleSave} disabled={!trained || !name.trim() || saving}>
                {saving ? 'Saving...' : 'Save'}
              </Button>
            </div>
            {!trained && (
              <p className="text-xs text-muted-foreground">
                Train the token tree first — its current state is what gets saved.
              </p>
            )}

            {trees.length === 0 ? (
              <div className="text-center py-6 text-sm text-muted-foreground">
                No saved trees yet. Train a token tree, then save it here to keep it across restarts.
              </div>
            ) : (
              <div className="divide-y divide-border/30">
                {trees.map(tree => (
                  <div key={tree.name} className="flex items-center gap-3 py-2">
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">{tree.name}</div>
                      <div className="flex flex-wrap items-center gap-2 mt-1">
                        <Chip label={`Vocab ${tree.vocab_size}`} />
                        <Chip label={`Merged ${tree.num_merges}`} />
                        {tree.trained && (
                          <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-success/10 text-success">
                            Trained
                          </span>
                        )}
                        <span className="text-[10px] text-muted-foreground">saved {formatDate(tree.saved_at)}</span>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleLoad(tree)}
                      disabled={busyName !== null}
                      aria-label={`Load ${tree.name}`}
                    >
                      {busyName === tree.name ? 'Loading...' : 'Load'}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive"
                      onClick={() => handleDelete(tree)}
                      disabled={busyName !== null}
                      aria-label={`Delete ${tree.name}`}
                    >
                      Delete
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
        <p className="text-xs text-muted-foreground">
          Saved trees persist to <span className="font-mono">data/token_trees/</span> and survive restarts.
          Loading one swaps it in as the current tree for the other explorer cards.
        </p>
      </CardContent>
    </Card>
  )
}
