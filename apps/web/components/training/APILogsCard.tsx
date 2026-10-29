'use client'

import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Progress } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { trainingController, type TrainFromSessionsParams } from '@/lib/training-controller'
import { soulsController } from '@/lib/souls-controller'
import { logger } from '@/lib/dev-log'

type Phase = 'idle' | 'extracting' | 'training' | 'complete' | 'error'

interface TrainResult {
  checkpoint?: string
  final_loss?: number
  num_pairs?: number
  perplexity?: number
  samples?: Array<{ prompt: string; response: string }>
}

interface FormConfig {
  epochs: number
  lr: number
  embed: number
  layers: number
  heads: number
}

const DEFAULTS: FormConfig = { epochs: 5, lr: 3e-4, embed: 128, layers: 4, heads: 4 }

function clamp(key: keyof FormConfig, v: number): number {
  const limits: Record<keyof FormConfig, [number, number]> = {
    epochs: [1, 100],
    lr: [1e-5, 1.0],
    embed: [16, 512],
    layers: [1, 12],
    heads: [1, 16],
  }
  const [min, max] = limits[key]
  return Math.min(max, Math.max(min, v))
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
  const [config, setConfig] = useState<FormConfig>(DEFAULTS)

  const [starting, setStarting] = useState(false)

  const start = useCallback(async () => {
    if (phase !== 'idle') return
    setStarting(true)
    setProgress(0)
    setLoss(null)
    setError(null)
    try {
      const params: TrainFromSessionsParams = {
        epochs: config.epochs,
        learning_rate: config.lr,
        n_embed: config.embed,
        n_layer: config.layers,
        n_head: config.heads,
        soul_name: 'api-logs-trained',
      }
      await trainingController.startFromSessionsSloNet(params)
      setPhase('training')

      for await (const event of trainingController.streamFromSessionsSloNet()) {
        const { phase: evPhase, status, data, meta, message } = event

        if (evPhase === 'PAIRS' && status === 'working') {
          setPhase('extracting')
        } else if (evPhase === 'TRAIN' && status === 'working') {
          setPhase('training')
          if (typeof data.loss === 'number') setLoss(data.loss)
          if (typeof data.step === 'number') {
            const total = typeof meta.total_steps === 'number' ? meta.total_steps : 1
            setProgress(Math.min(99, Math.round((data.step / Math.max(1, total)) * 100)))
          }
          if (typeof meta.epoch === 'number') setEpoch(meta.epoch as number)
          if (typeof meta.total_epochs === 'number') setTotalEpochs(meta.total_epochs as number)
        } else if (status === 'complete') {
          setProgress(100)
          setPhase('complete')
          setResult(data as TrainResult)
          addToast('Training complete!', 'success')
        } else if (status === 'error') {
          setPhase('error')
          setError(message || 'Training failed')
          addToast('Training failed', 'error')
        }
      }
    } catch (err: unknown) {
      setPhase('error')
      setError(err instanceof Error ? err.message : 'Unknown error')
      addToast('Training failed', 'error')
    } finally {
      setStarting(false)
    }
  }, [phase, config, addToast])

  const stop = useCallback(() => {
    trainingController.cancelFromSessionsSloNet().catch(e => {
      logger.error('Could not cancel training', { exception: String(e) })
      addToast('Could not cancel training', 'error')
    })
    setPhase('idle')
  }, [addToast])

  const loadForChat = useCallback(async () => {
    const raw = result?.checkpoint
    const name = raw?.split('/').pop()
    if (!name) {
      addToast('No checkpoint to load', 'error')
      return
    }
    setLoadingModel(true)
    try {
      await soulsController.loadCheckpoint(name)
      addToast(`Loaded: ${name}`, 'success')
    } catch {
      addToast('Could not load checkpoint', 'error')
    } finally {
      setLoadingModel(false)
    }
  }, [result, addToast])

  const setNum = useCallback(
    (key: keyof FormConfig) => (e: React.ChangeEvent<HTMLInputElement>) => {
      const raw = Number(e.target.value)
      if (!Number.isNaN(raw)) {
        setConfig(prev => ({ ...prev, [key]: clamp(key, raw) }))
      }
    },
    [],
  )

  const isRunning = phase === 'training' || phase === 'extracting'

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Train from API logs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {isRunning ? (
          <div className="space-y-2" aria-live="polite" aria-atomic="true">
            <Progress
              value={progress}
              max={100}
              label={phase === 'extracting' ? 'Extracting pairs...' : `Epoch ${epoch}/${totalEpochs}`}
              showValue
            />
            {loss != null && (
              <p className="text-[10px] text-muted-foreground/60 font-mono tabular-nums">Loss {loss.toFixed(4)}</p>
            )}
            <Button variant="destructive" size="sm" className="h-7 text-[11px]" onClick={stop}>
              Stop
            </Button>
          </div>
        ) : phase === 'complete' && result ? (
          <div className="space-y-2">
            <p className="text-[11px] text-success font-medium">Training complete</p>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground/60">
              {result.final_loss != null && (
                <>
                  <span>Final loss</span>
                  <span className="font-mono text-foreground tabular-nums">{result.final_loss.toFixed(4)}</span>
                </>
              )}
              {result.num_pairs != null && (
                <>
                  <span>Training pairs</span>
                  <span className="font-mono text-foreground tabular-nums">{result.num_pairs}</span>
                </>
              )}
              {result.perplexity != null && (
                <>
                  <span>Perplexity</span>
                  <span className="font-mono text-foreground tabular-nums">{result.perplexity.toFixed(2)}</span>
                </>
              )}
            </div>
            {result.samples && result.samples.length > 0 && (
              <div className="space-y-0.5">
                <p className="text-[10px] font-medium text-foreground">Samples</p>
                {result.samples.slice(0, 3).map((s, i) => (
                  <p key={i} className="text-[10px] text-muted-foreground/60 truncate">
                    {s.prompt} → {s.response}
                  </p>
                ))}
              </div>
            )}
            <div className="flex items-center gap-1.5">
              <Button size="sm" className="h-7 text-[11px]" onClick={loadForChat} disabled={loadingModel}>
                {loadingModel ? 'Loading...' : 'Load for chat'}
              </Button>
              <Button variant="outline" size="sm" className="h-7 text-[11px]" onClick={() => { setPhase('idle'); setError(null) }}>
                Train another
              </Button>
            </div>
          </div>
        ) : phase === 'error' ? (
          <StatusBanner
            variant="error"
            message={error || 'Training failed'}
            dismissible={false}
            onDismiss={() => setPhase('idle')}
          />
        ) : (
          <div className="space-y-3">
            <p className="text-[10px] text-muted-foreground/60">
              Train on your chat request/response logs. No dataset needed — uses data from your API sessions.
            </p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-epochs" variant="uppercase">Epochs</Label>
                <Input
                  id="api-epochs"
                  type="number"
                  min={1}
                  max={100}
                  value={config.epochs}
                  onChange={setNum('epochs')}
                  className="h-7 text-[11px] font-mono"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-lr" variant="uppercase">LR</Label>
                <Input
                  id="api-lr"
                  type="text"
                  inputMode="decimal"
                  value={config.lr}
                  onChange={setNum('lr')}
                  className="h-7 text-[11px] font-mono"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-embed" variant="uppercase">Embed</Label>
                <Input
                  id="api-embed"
                  type="number"
                  min={16}
                  max={512}
                  value={config.embed}
                  onChange={setNum('embed')}
                  className="h-7 text-[11px] font-mono"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-heads" variant="uppercase">Heads</Label>
                <Input
                  id="api-heads"
                  type="number"
                  min={1}
                  max={16}
                  value={config.heads}
                  onChange={setNum('heads')}
                  className="h-7 text-[11px] font-mono"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-layers" variant="uppercase">Layers</Label>
                <Input
                  id="api-layers"
                  type="number"
                  min={1}
                  max={12}
                  value={config.layers}
                  onChange={setNum('layers')}
                  className="h-7 text-[11px] font-mono"
                />
              </div>
            </div>
            <Button size="sm" className="h-7 text-[11px]" onClick={start} disabled={starting}>
              {starting ? 'Starting...' : 'Start training'}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
