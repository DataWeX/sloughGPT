'use client'

import { useState } from 'react'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { modelController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'

export function TurboTrainCard({
  datasets,
  session,
}: {
  datasets: UseTrainingDatasetsReturn
  session: UseTrainingSessionReturn
}) {
  const addToast = useToastStore(s => s.addToast)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [epochs, setEpochs] = useState(3)
  const [lr, setLr] = useState(3e-4)
  const [embed, setEmbed] = useState(128)
  const [heads, setHeads] = useState(4)
  const [layers, setLayers] = useState(3)

  const start = async () => {
    if (!datasets.selectedDataset) { addToast('Select a dataset first', 'error'); return }
    session.startTurboTrain(datasets.selectedDataset, {
      epochs, lr, embed, heads, layers,
    }, addToast)
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Turbo Train</CardTitle>
      </CardHeader>
      <CardContent>
        {session.turboPhase === 'training' ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
              </span>
              <span className="font-medium">Training encoder-decoder Transformer...</span>
            </div>
            <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-primary animate-pulse rounded-full" style={{ width: '60%' }} />
            </div>
            <p className="text-xs text-muted-foreground">Using our own torch shim — no PyTorch, no downloads</p>
          </div>
        ) : session.turboPhase === 'complete' && session.turboResult ? (
          <div className="rounded-lg border border-success/20 bg-success/5 p-4 space-y-3">
            <p className="text-sm font-medium text-success">Turbo training complete!</p>
            {session.turboResult!.final_loss != null && <p className="text-xs text-muted-foreground">Final loss: {session.turboResult!.final_loss.toFixed(4)}</p>}
            {session.turboResult!.total_steps != null && <p className="text-xs text-muted-foreground">Steps: {session.turboResult!.total_steps}</p>}
            {session.turboResult!.model_path && <p className="text-xs text-muted-foreground">Model: {session.turboResult!.model_path}</p>}
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => session.setTurboPhase('idle')}>Train another</Button>
              <Button size="sm" variant="ghost" onClick={async () => {
                if (session.turboResult!.model_path) {
                  try {
                    await modelController.loadModelPath(session.turboResult!.model_path)
                    addToast('Model loaded for chat', 'success')
                  } catch { addToast('Failed to load model', 'error') }
                }
              }}>Load for chat</Button>
            </div>
          </div>
        ) : session.turboPhase === 'error' ? (
          <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-4 space-y-3">
            <p className="text-sm font-medium text-destructive">Training failed</p>
            <p className="text-xs text-muted-foreground">{session.turboError || 'Unknown error'}</p>
            <Button size="sm" variant="outline" onClick={() => session.setTurboPhase('idle')}>Dismiss</Button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              {datasets.datasets.length === 0 ? (
                <>
                  <span className="text-xs text-muted-foreground">No datasets yet — import one to use Turbo Train.</span>
                  <Button size="sm" variant="outline" onClick={() => datasets.setImportModalOpen(true)}>+ Import</Button>
                </>
              ) : (
                <>
                  <Select value={datasets.selectedDataset} onValueChange={datasets.setSelectedDataset}>
                    <SelectTrigger className="h-8 text-xs font-mono flex-1 max-w-xs" aria-label="Dataset for turbo training">
                      <SelectValue placeholder="Select dataset..." />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets.datasets.map(ds => <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <Button size="sm" onClick={start} disabled={!datasets.selectedDataset || datasets.datasets.length === 0}>
                    Train with Turbo
                  </Button>
                </>
              )}
            </div>
            <p className="text-[11px] text-muted-foreground">
              Encoder-decoder Transformer via our own torch shim. No PyTorch, no downloads, runs on CPU.
              Saves model to <code className="text-xs">models/turbo-trained/</code>
            </p>
            <div className="border-t border-border/40 pt-2">
              <button className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                onClick={() => setShowAdvanced(!showAdvanced)}>
                {showAdvanced ? 'Hide' : 'Show'} advanced settings
              </button>
              {showAdvanced && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">Epochs</label>
                    <input type="number" value={epochs} onChange={e => setEpochs(Number(e.target.value))}
                      className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" min={1} max={100} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">Learning rate</label>
                    <input type="number" value={lr} onChange={e => setLr(Number(e.target.value))}
                      className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" step={1e-5} min={1e-5} max={1} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">Embed dim</label>
                    <input type="number" value={embed} onChange={e => setEmbed(Number(e.target.value))}
                      className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" min={16} max={1024} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">Heads</label>
                    <input type="number" value={heads} onChange={e => setHeads(Number(e.target.value))}
                      className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" min={1} max={64} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs text-muted-foreground">Encoder/Decoder layers</label>
                    <input type="number" value={layers} onChange={e => setLayers(Number(e.target.value))}
                      className="w-full h-8 rounded border border-border/50 bg-background px-2 text-xs" min={1} max={24} />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
