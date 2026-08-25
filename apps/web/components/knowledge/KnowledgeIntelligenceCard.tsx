'use client'

import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Textarea, Progress } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

export function KnowledgeIntelligenceCard() {
  const addToast = useToastStore(s => s.addToast)

  const [labelText, setLabelText] = useState('')
  const [labelResult, setLabelResult] = useState<{ label: string; confidence: number; reason: string; scores: Record<string, number> } | null>(null)
  const [labelLoading, setLabelLoading] = useState(false)

  const [dupText, setDupText] = useState('')
  const [dupResult, setDupResult] = useState<{ is_duplicate: boolean; best_match: string | null; score: number; threshold: number } | null>(null)
  const [dupLoading, setDupLoading] = useState(false)

  const [catText, setCatText] = useState('')
  const [catResult, setCatResult] = useState<{ topic: string; suggestions: Array<{ topic: string; score: number }> } | null>(null)
  const [catLoading, setCatLoading] = useState(false)

  const [embedderStatus, setEmbedderStatus] = useState<{ trained: boolean; info: { embed_dim: number; vocab_size: number; path: string } | null } | null>(null)
  const [trainingEmbedder, setTrainingEmbedder] = useState(false)
  const [embedderLoading, setEmbedderLoading] = useState(false)

  const [gaps, setGaps] = useState<{ gaps: Array<{ topic: string; suggestion: string }>; total_facts: number; topics: string[] } | null>(null)
  const [gapsLoading, setGapsLoading] = useState(false)

  const handleLabel = useCallback(async () => {
    if (!labelText.trim()) return
    setLabelLoading(true)
    try {
      const result = await knowledgeController.label(labelText)
      setLabelResult(result)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Labeling failed'), 'error')
    } finally {
      setLabelLoading(false)
    }
  }, [labelText, addToast])

  const handleCheckDuplicate = useCallback(async () => {
    if (!dupText.trim()) return
    setDupLoading(true)
    try {
      const result = await knowledgeController.checkDuplicate(dupText)
      setDupResult(result)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Duplicate check failed'), 'error')
    } finally {
      setDupLoading(false)
    }
  }, [dupText, addToast])

  const handleCategorize = useCallback(async () => {
    if (!catText.trim()) return
    setCatLoading(true)
    try {
      const result = await knowledgeController.categorize(catText)
      setCatResult(result)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Categorization failed'), 'error')
    } finally {
      setCatLoading(false)
    }
  }, [catText, addToast])

  const handleCheckEmbedder = useCallback(async () => {
    setEmbedderLoading(true)
    try {
      const result = await knowledgeController.getEmbedderStatus()
      setEmbedderStatus(result)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not check embedder'), 'error')
    } finally {
      setEmbedderLoading(false)
    }
  }, [addToast])

  const handleTrainEmbedder = useCallback(async () => {
    setTrainingEmbedder(true)
    try {
      const result = await knowledgeController.trainEmbedder()
      addToast(`Embedder trained on ${result.texts_used} texts (loss: ${result.final_loss.toFixed(4)})`, 'success')
      await handleCheckEmbedder()
    } catch (err) {
      addToast(extractErrorMessage(err, 'Embedder training failed'), 'error')
    } finally {
      setTrainingEmbedder(false)
    }
  }, [addToast, handleCheckEmbedder])

  const handleGaps = useCallback(async () => {
    setGapsLoading(true)
    try {
      const result = await knowledgeController.gaps()
      setGaps(result)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Gap analysis failed'), 'error')
    } finally {
      setGapsLoading(false)
    }
  }, [addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Knowledge Intelligence</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Auto-label */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Auto-label</p>
          <div className="flex gap-2">
            <Input
              value={labelText}
              onChange={e => setLabelText(e.target.value)}
              placeholder="Enter text to classify..."
              className="h-8 text-xs"
              onKeyDown={e => e.key === 'Enter' && void handleLabel()}
            />
            <Button size="sm" onClick={() => void handleLabel()} disabled={labelLoading || !labelText.trim()} className="shrink-0">
              {labelLoading ? '...' : 'Label'}
            </Button>
          </div>
          {labelResult && (
            <div className="rounded bg-muted/30 px-3 py-2 text-xs space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-medium">Label: {labelResult.label}</span>
                <span className="text-muted-foreground">({(labelResult.confidence * 100).toFixed(0)}%)</span>
              </div>
              {labelResult.reason && <p className="text-muted-foreground">{labelResult.reason}</p>}
              {Object.keys(labelResult.scores).length > 0 && (
                <div className="flex flex-wrap gap-1 pt-1">
                  {Object.entries(labelResult.scores)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([k, v]) => (
                      <span key={k} className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
                        {k}: {(v * 100).toFixed(0)}%
                      </span>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="h-px bg-border/50" />

        {/* Duplicate check */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Duplicate check</p>
          <div className="flex gap-2">
            <Input
              value={dupText}
              onChange={e => setDupText(e.target.value)}
              placeholder="Enter text to check for duplicates..."
              className="h-8 text-xs"
              onKeyDown={e => e.key === 'Enter' && void handleCheckDuplicate()}
            />
            <Button size="sm" onClick={() => void handleCheckDuplicate()} disabled={dupLoading || !dupText.trim()} className="shrink-0">
              {dupLoading ? '...' : 'Check'}
            </Button>
          </div>
          {dupResult && (
            <div className={`rounded px-3 py-2 text-xs ${dupResult.is_duplicate ? 'bg-warning/10' : 'bg-success/10'}`}>
              {dupResult.is_duplicate ? (
                <div className="space-y-1">
                  <p className="font-medium text-warning">Duplicate found</p>
                  {dupResult.best_match && <p className="text-muted-foreground truncate">Match: {dupResult.best_match}</p>}
                  <p className="text-muted-foreground">Score: {(dupResult.score * 100).toFixed(0)}% (threshold: {(dupResult.threshold * 100).toFixed(0)}%)</p>
                </div>
              ) : (
                <p className="font-medium text-success">No duplicates found</p>
              )}
            </div>
          )}
        </div>

        <div className="h-px bg-border/50" />

        {/* Categorize */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Auto-categorize</p>
          <div className="flex gap-2">
            <Input
              value={catText}
              onChange={e => setCatText(e.target.value)}
              placeholder="Enter text to categorize..."
              className="h-8 text-xs"
              onKeyDown={e => e.key === 'Enter' && void handleCategorize()}
            />
            <Button size="sm" onClick={() => void handleCategorize()} disabled={catLoading || !catText.trim()} className="shrink-0">
              {catLoading ? '...' : 'Categorize'}
            </Button>
          </div>
          {catResult && (
            <div className="rounded bg-muted/30 px-3 py-2 text-xs space-y-1">
              <p><span className="font-medium">Topic:</span> {catResult.topic}</p>
              {catResult.suggestions.length > 1 && (
                <div className="flex flex-wrap gap-1 pt-1">
                  {catResult.suggestions.slice(0, 5).map(s => (
                    <span key={s.topic} className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
                      {s.topic}: {(s.score * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="h-px bg-border/50" />

        {/* Embedder */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium text-muted-foreground">Embedder</p>
            <div className="flex gap-1">
              <Button size="sm" variant="ghost" onClick={() => void handleCheckEmbedder()} disabled={embedderLoading}>
                <IconRefresh className={`h-3 w-3 ${embedderLoading ? 'animate-spin' : ''}`} />
              </Button>
              <Button size="sm" variant="outline" onClick={() => void handleTrainEmbedder()} disabled={trainingEmbedder}>
                {trainingEmbedder ? 'Training...' : 'Train'}
              </Button>
            </div>
          </div>
          {embedderStatus && (
            <div className="rounded bg-muted/30 px-3 py-2 text-xs space-y-1">
              <div className="flex items-center gap-2">
                <span className={`inline-block h-2 w-2 rounded-full ${embedderStatus.trained ? 'bg-success' : 'bg-muted-foreground'}`} />
                <span>{embedderStatus.trained ? 'Trained' : 'Not trained'}</span>
              </div>
              {embedderStatus.info && (
                <p className="text-muted-foreground">
                  dim={embedderStatus.info.embed_dim}, vocab={embedderStatus.info.vocab_size}
                </p>
              )}
            </div>
          )}
        </div>

        <div className="h-px bg-border/50" />

        {/* Knowledge gaps */}
        <div className="space-y-2">
          <Button size="sm" variant="outline" onClick={() => void handleGaps()} disabled={gapsLoading} className="w-full">
            {gapsLoading ? 'Analyzing...' : 'Analyze Knowledge Gaps'}
          </Button>
          {gaps && (
            <div className="space-y-2">
              <div className="flex gap-3 text-xs">
                <span className="text-muted-foreground">Facts: {gaps.total_facts}</span>
                <span className="text-muted-foreground">Topics: {gaps.topics.length}</span>
                <span className="text-muted-foreground">Gaps: {gaps.gaps.length}</span>
              </div>
              {gaps.gaps.length > 0 && (
                <div className="rounded bg-warning/10 px-3 py-2 text-xs space-y-1">
                  {gaps.gaps.map((g, i) => (
                    <div key={i}>
                      <span className="font-medium">{g.topic}:</span> <span className="text-muted-foreground">{g.suggestion}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
