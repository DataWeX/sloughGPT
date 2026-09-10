'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Progress } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { toIPA } from '@/lib/phoneme-controller'

interface PhonemeComparisonProps {
  target: string
  spoken: string
  targetPhonemes: string[]
  spokenPhonemes: string[]
  score: number
  precision: number
  recall: number
}

interface PhonemePair {
  target: string | null
  spoken: string | null
  match: boolean
  targetIPA: string
  spokenIPA: string
}

function alignPhonemes(target: string[], spoken: string[]): PhonemePair[] {
  const pairs: PhonemePair[] = []
  const maxLen = Math.max(target.length, spoken.length)

  for (let i = 0; i < maxLen; i++) {
    const t = target[i] ?? null
    const s = spoken[i] ?? null
    pairs.push({
      target: t,
      spoken: s,
      match: t !== null && s !== null && t === s,
      targetIPA: t ? toIPA([t])[0] : '',
      spokenIPA: s ? toIPA([s])[0] : '',
    })
  }

  return pairs
}

function getScoreColor(score: number): string {
  if (score >= 0.8) return 'text-success'
  if (score >= 0.5) return 'text-warning'
  return 'text-destructive'
}

function getScoreLabel(score: number): string {
  if (score >= 0.9) return 'Excellent'
  if (score >= 0.8) return 'Great'
  if (score >= 0.7) return 'Good'
  if (score >= 0.5) return 'Fair'
  return 'Needs Work'
}

export default function PhonemeComparison({
  target,
  spoken,
  targetPhonemes,
  spokenPhonemes,
  score,
  precision,
  recall,
}: PhonemeComparisonProps) {
  const pairs = useMemo(
    () => alignPhonemes(targetPhonemes, spokenPhonemes),
    [targetPhonemes, spokenPhonemes]
  )

  const matchCount = pairs.filter(p => p.match).length
  const totalCount = pairs.length

  const mismatchedPhonemes = useMemo(() => {
    return pairs
      .filter(p => !p.match && p.target && p.spoken)
      .map(p => ({
        target: p.target!,
        spoken: p.spoken!,
        targetIPA: p.targetIPA,
        spokenIPA: p.spokenIPA,
      }))
  }, [pairs])

  const missingPhonemes = useMemo(() => {
    return pairs.filter(p => p.target && !p.spoken).map(p => p.target!)
  }, [pairs])

  const extraPhonemes = useMemo(() => {
    return pairs.filter(p => !p.target && p.spoken).map(p => p.spoken!)
  }, [pairs])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Analysis</span>
          <span className={`text-lg font-bold ${getScoreColor(score)}`}>
            {getScoreLabel(score)}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className={`text-2xl font-bold ${getScoreColor(score)}`}>
              {(score * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-muted-foreground">Overall Score</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{(precision * 100).toFixed(0)}%</p>
            <p className="text-xs text-muted-foreground">Precision</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{(recall * 100).toFixed(0)}%</p>
            <p className="text-xs text-muted-foreground">Recall</p>
          </div>
        </div>

        <Progress value={score * 100} />

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Words Compared</span>
            <span className="font-medium">{target} → {spoken}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Phoneme Match</span>
            <span className="font-medium">{matchCount}/{totalCount}</span>
          </div>
        </div>

        <Collapsible>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
              <span className="font-medium">Phoneme-by-Phoneme Comparison</span>
              <span className="text-muted-foreground">{matchCount}/{totalCount} matched</span>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="p-2 space-y-1">
              {pairs.map((pair, i) => (
                <div
                  key={i}
                  className={`flex items-center gap-3 p-2 rounded text-sm ${
                    pair.match
                      ? 'bg-success/10'
                      : pair.target && pair.spoken
                      ? 'bg-destructive/10'
                      : 'bg-warning/10'
                  }`}
                >
                  <span className="w-8 text-center text-muted-foreground">{i + 1}</span>
                  <span className="w-16 font-mono text-center">
                    {pair.target ? (
                      <>
                        <span className="font-medium">{pair.target}</span>
                        <span className="text-xs text-muted-foreground ml-1">{pair.targetIPA}</span>
                      </>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </span>
                  <span className="text-muted-foreground">→</span>
                  <span className="w-16 font-mono text-center">
                    {pair.spoken ? (
                      <>
                        <span className="font-medium">{pair.spoken}</span>
                        <span className="text-xs text-muted-foreground ml-1">{pair.spokenIPA}</span>
                      </>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </span>
                  <Badge
                    variant={pair.match ? 'default' : 'destructive'}
                    className="ml-auto"
                  >
                    {pair.match ? '✓' : '✗'}
                  </Badge>
                </div>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>

        {mismatchedPhonemes.length > 0 && (
          <Collapsible>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
                <span className="font-medium">Mismatched Phonemes</span>
                <Badge variant="destructive">{mismatchedPhonemes.length}</Badge>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-2 space-y-1">
                {mismatchedPhonemes.map((m, i) => (
                  <div key={i} className="flex items-center justify-between p-2 rounded bg-destructive/5 text-sm">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{m.target}</Badge>
                      <span className="text-muted-foreground">→</span>
                      <Badge variant="outline">{m.spoken}</Badge>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {m.targetIPA} → {m.spokenIPA}
                    </span>
                  </div>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {missingPhonemes.length > 0 && (
          <div className="p-2 rounded bg-warning/10 text-sm">
            <span className="font-medium">Missing:</span>{' '}
            {missingPhonemes.map((p, i) => (
              <Badge key={i} variant="outline" className="mr-1">{p}</Badge>
            ))}
          </div>
        )}

        {extraPhonemes.length > 0 && (
          <div className="p-2 rounded bg-warning/10 text-sm">
            <span className="font-medium">Extra:</span>{' '}
            {extraPhonemes.map((p, i) => (
              <Badge key={i} variant="outline" className="mr-1">{p}</Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
