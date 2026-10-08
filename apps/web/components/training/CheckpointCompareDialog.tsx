'use client'

import { useEffect, useState } from 'react'
import {
  Button,
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  Textarea,
} from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { trainingJobsController } from '@/lib/training-controller'
import { formatToastError } from '@/lib/error-utils'

export interface CompareSide {
  name: string
  text: string
}

interface CheckpointCompareDialogProps {
  open: boolean
  onClose: () => void
  // Rows, not bare names: the selects are valued by path so two same-named
  // twins stay distinct choices, labelled with the short id like the list.
  checkpoints: { name: string; path: string; integrity_hash?: string }[]
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export function CheckpointCompareDialog({
  open,
  onClose,
  checkpoints,
  addToast,
}: CheckpointCompareDialogProps) {
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ a: CompareSide; b: CompareSide } | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    setA((prev) => prev || checkpoints[0]?.path || '')
    setB((prev) => prev || checkpoints[1]?.path || checkpoints[0]?.path || '')
  }, [open, checkpoints])

  const rowA = checkpoints.find((cp) => cp.path === a)
  const rowB = checkpoints.find((cp) => cp.path === b)

  // Same question, different weights — picking one checkpoint twice would
  // just print the same answer twice.
  const canRun = Boolean(rowA && rowB && rowA.path !== rowB.path && prompt.trim() && !loading)

  const handleRun = async () => {
    if (!canRun || !rowA || !rowB) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      // Each side carries its row's address, so compare runs the exact
      // files shown even when names collide across roots.
      const res = await trainingJobsController.compareCheckpoints(
        rowA.name,
        rowB.name,
        prompt.trim(),
        { pathA: rowA.path, pathB: rowB.path },
      )
      setResult(res)
    } catch (e) {
      const msg = formatToastError(e, 'Could not compare checkpoints')
      setError(msg)
      addToast(msg, 'error')
    } finally {
      setLoading(false)
    }
  }

  if (!open) return null

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose()
      }}
    >
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-sm">Compare checkpoints</DialogTitle>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-2">
          <div className="space-y-1">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
              Checkpoint A
            </p>
            <select
              className="w-full text-xs border border-border/60 rounded-md p-1.5 bg-background"
              value={a}
              onChange={(e) => setA(e.target.value)}
              aria-label="Checkpoint A"
            >
              {checkpoints.map((cp) => (
                <option key={cp.path} value={cp.path}>
                  {cp.name}
                  {cp.integrity_hash ? ` · id ${cp.integrity_hash.slice(0, 8)}` : ''}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
              Checkpoint B
            </p>
            <select
              className="w-full text-xs border border-border/60 rounded-md p-1.5 bg-background"
              value={b}
              onChange={(e) => setB(e.target.value)}
              aria-label="Checkpoint B"
            >
              {checkpoints.map((cp) => (
                <option key={cp.path} value={cp.path}>
                  {cp.name}
                  {cp.integrity_hash ? ` · id ${cp.integrity_hash.slice(0, 8)}` : ''}
                </option>
              ))}
            </select>
          </div>
        </div>

        <Textarea
          aria-label="Comparison prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask both models the same question..."
          rows={3}
          className="text-[11px] font-mono resize-none"
        />

        <div className="flex items-center gap-2">
          {a === b && (
            <span className="text-[10px] text-muted-foreground/60">
              Pick two different checkpoints
            </span>
          )}
          <div className="flex gap-1.5 ml-auto">
            <Button size="sm" className="h-7 text-[11px]" onClick={handleRun} disabled={!canRun}>
              {loading ? 'Comparing...' : 'Compare'}
            </Button>
            <Button size="sm" variant="ghost" className="h-7 text-[11px]" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>

        {error && <StatusBanner variant="error" message={error} dismissible={false} />}

        {result ? (
          <div className="grid grid-cols-2 gap-2">
            {(
              [
                ['A', result.a],
                ['B', result.b],
              ] as const
            ).map(([label, side]) => (
              <div
                key={label}
                className="rounded-lg border border-border/50 bg-muted/20 p-2.5 space-y-1"
              >
                <p className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">
                  {label} · {side.name}
                </p>
                <p className="text-[11px] font-mono whitespace-pre-wrap text-foreground">
                  {side.text}
                </p>
              </div>
            ))}
          </div>
        ) : (
          !error && (
            <p className="text-[10px] text-muted-foreground/60">
              Run a comparison to see both answers side by side.
            </p>
          )
        )}
      </DialogContent>
    </Dialog>
  )
}
