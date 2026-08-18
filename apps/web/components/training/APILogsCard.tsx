'use client'

import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Progress } from '@sloughgpt/strui'
import { trainingController } from '@/lib/training-controller'

type Phase = 'idle' | 'extracting' | 'training' | 'complete' | 'error'

interface TrainResult {
  checkpoint?: string
  final_loss?: number
  num_pairs?: number
  perplexity?: number
  samples?: Array<{ prompt: string; response: string }>
}

function isTrainResult(data: Record<string, unknown>): data is Record<string, unknown> & TrainResult {
  return data != null && typeof data === 'object'
}

export function APILogsCard({
  addToast,
}: {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}) {
  const [phase, setPhase] = useState<Phase>('idle')
  const [progress, setProgress] = useState(0)
  const [loss, setLoss] = useState<number | null>(null)
  const [epoch, setEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(0)
  const [result, setResult] = useState<TrainResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loadingModel, setLoadingModel] = useState(false)
  const [config, setConfig] = useState({ epochs: 5, lr: 3e-4, embed: 128, layers: 4, heads: 4 })

  const start = useCallback(async () => {
    setPhase('extracting')
    setProgress(0)
    setLoss(null)
    setEpoch(0)
    setTotalEpochs(config.epochs)
    setResult(null)
    setError(null)

    try {
      // Start training
      await trainingController.startFromSessionsSloNet({
        epochs: config.epochs,
        learning_rate: config.lr,
        n_embed: config.embed,
        n_layer: config.layers,
        n_head: config.heads,
        soul_name: 'api-logs-trained',
      })

      setPhase('training')

      // Stream SSE progress via controller
      for await (const event of trainingController.streamFromSessionsSloNet()) {
        const { phase: phaseEv, status, data, meta, message } = event

        if (phaseEv === 'PAIRS' && status === 'working') {
          setPhase('extracting')
        } else if (phaseEv === 'TRAIN' && status === 'working') {
          setPhase('training')
          if (typeof data.loss === 'number') setLoss(data.loss)
          if (typeof data.step === 'number') setProgress(Math.min(99, Math.round((data.step / Math.max(1, (meta?.total_steps as number) || 1)) * 100)))
          if (typeof meta?.epoch === 'number') setEpoch(meta.epoch as number)
          if (typeof meta?.total_epochs === 'number') setTotalEpochs(meta.total_epochs as number)
        } else if (status === 'complete') {
          setProgress(100)
          setPhase('complete')
          setResult(isTrainResult(data) ? data : {})
          addToast('Training complete!', 'success')
        } else if (status === 'error') {
          setPhase('error')
          setError(message || 'Unknown error')
          addToast('Training failed', 'error')
        }
      }
    } catch (err: unknown) {
      setPhase('error')
      setError(err instanceof Error ? err.message : 'Unknown error')
      addToast('Training failed', 'error')
    }
  }, [config, addToast])

  const stop = useCallback(() => {
    trainingController.cancelFromSessionsSloNet().catch(() => {})
    setPhase('idle')
  }, [])

  const loadForChat = useCallback(async () => {
    const name = result?.checkpoint?.split('/').pop()
    if (!name) {
      addToast('No checkpoint to load', 'error')
      return
    }
    setLoadingModel(true)
    try {
      await trainingController.loadCheckpoint(name)
      addToast(`Loaded: ${name}`, 'success')
    } catch {
      addToast('Failed to load checkpoint', 'error')
    } finally {
      setLoadingModel(false)
    }
  }, [result, addToast])

  const setNum = (key: keyof typeof config) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setConfig(prev => ({ ...prev, [key]: Number(e.target.value) }))

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Train from API logs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {phase === 'training' || phase === 'extracting' ? (
          <div className="space-y-3" aria-live="polite">
            <Progress value={progress} max={100} />
            <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-3">
              <span>{phase === 'extracting' ? 'Extracting pairs...' : `Epoch ${epoch}/${totalEpochs}`}</span>
              {loss != null && <span>Loss {loss.toFixed(4)}</span>}
              <span>{progress}%</span>
            </div>
            <Button variant="destructive" size="sm" onClick={stop}>
              Stop
            </Button>
          </div>
        ) : phase === 'complete' && result ? (
          <div className="space-y-2 text-sm">
            <p className="text-success">Training complete!</p>
            <div className="space-y-1 text-xs text-muted-foreground">
              {result.final_loss != null && <p>Final loss: {result.final_loss.toFixed(4)}</p>}
              {result.num_pairs != null && <p>Training pairs: {result.num_pairs}</p>}
              {result.perplexity != null && <p>Perplexity: {result.perplexity.toFixed(2)}</p>}
              {result.samples && result.samples.length > 0 && (
                <div className="mt-2 space-y-1">
                  <p className="font-medium text-foreground">Samples:</p>
                  {result.samples.slice(0, 3).map((s, i) => (
                    <p key={i} className="truncate">{s.prompt} → {s.response}</p>
                  ))}
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={loadForChat} disabled={loadingModel}>
                {loadingModel ? 'Loading...' : 'Load for chat'}
              </Button>
              <Button variant="outline" size="sm" onClick={() => setPhase('idle')}>
                Train another
              </Button>
            </div>
          </div>
        ) : phase === 'error' ? (
          <div className="space-y-2 text-sm">
            <p className="text-destructive">Training failed</p>
            {error && <p className="text-xs text-muted-foreground">{error}</p>}
            <Button variant="outline" size="sm" onClick={() => setPhase('idle')}>
              Dismiss
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Train a model on your chat request/response logs. No dataset needed — uses data from your API sessions.
            </p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-epochs" variant="uppercase">Epochs</Label>
                <Input id="api-epochs" type="number" min={1} max={100} value={config.epochs}
                  onChange={setNum('epochs')} className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-lr" variant="uppercase">LR</Label>
                <Input id="api-lr" type="text" inputMode="decimal" value={config.lr}
                  onChange={setNum('lr')} className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-embed" variant="uppercase">Embed</Label>
                <Input id="api-embed" type="number" min={16} max={512} value={config.embed}
                  onChange={setNum('embed')} className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-heads" variant="uppercase">Heads</Label>
                <Input id="api-heads" type="number" min={1} max={16} value={config.heads}
                  onChange={setNum('heads')} className="h-8 text-xs font-mono" />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-layers" variant="uppercase">Layers</Label>
                <Input id="api-layers" type="number" min={1} max={12} value={config.layers}
                  onChange={setNum('layers')} className="h-8 text-xs font-mono" />
              </div>
            </div>
            <Button size="sm" onClick={start}>
              Start training
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
