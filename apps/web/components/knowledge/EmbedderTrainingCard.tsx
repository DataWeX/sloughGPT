'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'

export function EmbedderTrainingCard() {
  const [training, setTraining] = useState(false)
  const [embedderStatus, setEmbedderStatus] = useState<{ trained: boolean; info: { embed_dim: number; vocab_size: number } | null } | null>(null)
  const [result, setResult] = useState<{ texts_used: number; epochs: number; final_loss: number } | null>(null)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    knowledgeController.getEmbedderStatus().then(setEmbedderStatus).catch(() => {})
  }, [])

  const handleTrain = async () => {
    setTraining(true)
    try {
      const res = await knowledgeController.trainEmbedder()
      setResult(res)
      setEmbedderStatus({ trained: true, info: { embed_dim: 64, vocab_size: 1024 } })
      addToast(`Embedder trained on ${res.texts_used} texts in ${res.epochs} epochs`, 'success')
    } catch { addToast('Training failed', 'error') }
    setTraining(false)
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Text Embedder</CardTitle>
        <Button size="sm" onClick={handleTrain} disabled={training}>
          {training ? 'Training\u2026' : embedderStatus?.trained ? 'Retrain' : 'Train Embedder'}
        </Button>
      </CardHeader>
      <CardContent>
        {embedderStatus === null ? (
          <div className="h-8 animate-pulse bg-muted rounded" />
        ) : embedderStatus.trained ? (
          <div className="flex items-center gap-4 text-sm flex-wrap">
            <Badge variant="default" label="Trained" />
            <span className="text-muted-foreground">
              {embedderStatus.info?.embed_dim}d, {embedderStatus.info?.vocab_size} vocab
            </span>
            {result && (
              <span className="text-muted-foreground">
                loss {result.final_loss.toFixed(4)} on {result.texts_used} texts
              </span>
            )}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            Train a SloNet text embedder on your knowledge + datasets for better semantic search. No downloads required.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
