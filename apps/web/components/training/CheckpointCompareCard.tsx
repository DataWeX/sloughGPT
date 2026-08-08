'use client'

import { useCallback, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface CheckpointCompareCardProps {
  checkpoints: Checkpoint[]
}

function fmtDuration(s?: number | null): string {
  if (s == null) return '—'
  if (s < 60) return `${s.toFixed(0)}s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

function fmtLoss(v?: number | null): string {
  if (v == null) return '—'
  return v.toFixed(4)
}

interface CompareRowProps {
  label: string
  a: string
  b: string
  lowerBetter?: boolean
  valA?: number | null
  valB?: number | null
}

function CompareRow({ label, a, b, lowerBetter, valA, valB }: CompareRowProps) {
  let winner: 'a' | 'b' | 'tie' = 'tie'
  if (valA != null && valB != null && valA !== valB) {
    winner = lowerBetter ? (valA < valB ? 'a' : 'b') : (valA > valB ? 'a' : 'b')
  }
  return (
    <div className="grid grid-cols-3 gap-2 text-[11px] py-1 border-b border-border/30 last:border-0">
      <span className="text-muted-foreground/70">{label}</span>
      <span className={`font-mono text-center ${winner === 'a' ? 'text-success font-medium' : ''}`}>{a}</span>
      <span className={`font-mono text-center ${winner === 'b' ? 'text-success font-medium' : ''}`}>{b}</span>
    </div>
  )
}

export function CheckpointCompareCard({ checkpoints }: CheckpointCompareCardProps) {
  const [pickA, setPickA] = useState<string>('')
  const [pickB, setPickB] = useState<string>('')
  const [expanded, setExpanded] = useState(false)

  const cpA = checkpoints.find(c => c.name === pickA)
  const cpB = checkpoints.find(c => c.name === pickB)

  const canCompare = cpA && cpB && cpA.name !== cpB.name

  const handleSwap = useCallback(() => {
    setPickA(pickB)
    setPickB(pickA)
  }, [pickA, pickB])

  if (checkpoints.length < 2) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Compare checkpoints</CardTitle>
        <Button
          size="sm"
          variant="ghost"
          className="h-6 text-[11px]"
          onClick={() => setExpanded(!expanded)}
          aria-label="Toggle comparison"
          data-testid="toggle-compare"
        >
          {expanded ? 'Hide' : 'Compare'}
        </Button>
      </CardHeader>
      {expanded && (
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 block">Checkpoint A</label>
              <Select value={pickA} onValueChange={setPickA}>
                <SelectTrigger className="h-7 text-[11px]" aria-label="Select first checkpoint">
                  <SelectValue placeholder="Select..." />
                </SelectTrigger>
                <SelectContent>
                  {checkpoints.map(cp => (
                    <SelectItem key={cp.name} value={cp.name}>{cp.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 block">Checkpoint B</label>
              <Select value={pickB} onValueChange={setPickB}>
                <SelectTrigger className="h-7 text-[11px]" aria-label="Select second checkpoint">
                  <SelectValue placeholder="Select..." />
                </SelectTrigger>
                <SelectContent>
                  {checkpoints.map(cp => (
                    <SelectItem key={cp.name} value={cp.name}>{cp.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {canCompare && (
            <>
              <button className="text-[10px] text-muted-foreground hover:text-foreground" onClick={handleSwap}>
                Swap A/B
              </button>
              <div className="space-y-0">
                <div className="grid grid-cols-3 gap-2 text-[10px] text-muted-foreground/50 uppercase tracking-wider pb-1 border-b border-border/30">
                  <span></span>
                  <span className="text-center">A</span>
                  <span className="text-center">B</span>
                </div>
                <CompareRow label="Loss" a={fmtLoss(cpA.loss)} b={fmtLoss(cpB.loss)} lowerBetter valA={cpA.loss} valB={cpB.loss} />
                <CompareRow label="Epochs" a={String(cpA.epochs_trained ?? '—')} b={String(cpB.epochs_trained ?? '—')} lowerBetter={false} valA={cpA.epochs_trained} valB={cpB.epochs_trained} />
                <CompareRow label="Duration" a={fmtDuration(cpA.training_duration_s)} b={fmtDuration(cpB.training_duration_s)} lowerBetter valA={cpA.training_duration_s} valB={cpB.training_duration_s} />
                <CompareRow label="Vocab size" a={String(cpA.vocab_size ?? '—')} b={String(cpB.vocab_size ?? '—')} lowerBetter={false} valA={cpA.vocab_size} valB={cpB.vocab_size} />
                <CompareRow label="Size" a={cpA.size_mb != null ? `${cpA.size_mb} MB` : '—'} b={cpB.size_mb != null ? `${cpB.size_mb} MB` : '—'} lowerBetter valA={cpA.size_mb} valB={cpB.size_mb} />
                <CompareRow label="Type" a={cpA.model_type ?? '—'} b={cpB.model_type ?? '—'} />
                <CompareRow label="Dataset" a={cpA.training_dataset ?? '—'} b={cpB.training_dataset ?? '—'} />
              </div>

              {cpA.traits && cpB.traits && Object.keys(cpA.traits).length > 0 && Object.keys(cpB.traits).length > 0 && (
                <div className="space-y-1">
                  <p className="text-[10px] text-muted-foreground/50 uppercase tracking-wider">Personality traits</p>
                  {Object.keys(cpA.traits).map(trait => (
                    <CompareRow
                      key={trait}
                      label={trait}
                      a={cpA.traits![trait]?.toFixed(2) ?? '—'}
                      b={cpB.traits![trait]?.toFixed(2) ?? '—'}
                      lowerBetter={false}
                      valA={cpA.traits![trait]}
                      valB={cpB.traits![trait]}
                    />
                  ))}
                </div>
              )}
            </>
          )}

          {!canCompare && pickA && pickB && pickA === pickB && (
            <p className="text-[11px] text-muted-foreground/50">Select two different checkpoints to compare.</p>
          )}
        </CardContent>
      )}
    </Card>
  )
}
