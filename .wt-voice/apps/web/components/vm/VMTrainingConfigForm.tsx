'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

interface TrainConfig {
  dataset: string
  epochs: number
  lr: number
  batch_size: number
  n_layer: number
  n_head: number
  embed_dim: number
}

interface ConfigHint {
  label: string
  message: string
}

interface VMTrainingConfigFormProps {
  config: TrainConfig
  onConfigChange: (config: TrainConfig) => void
  datasetNames: string[]
  role: string
  onRoleChange: (role: string) => void
  hints: ConfigHint[]
  onReset: () => void
  onLoadSample: () => void
  onLaunch: () => void
  launchedJob: number | null
  onDismissJob: () => void
}

const num = (v: string) => (v.trim() === '' ? NaN : Number(v))

export function VMTrainingConfigForm({
  config,
  onConfigChange,
  datasetNames,
  role,
  onRoleChange,
  hints,
  onReset,
  onLoadSample,
  onLaunch,
  launchedJob,
  onDismissJob,
}: VMTrainingConfigFormProps) {
  const [customDataset, setCustomDataset] = useState(false)

  const updateConfig = (patch: Partial<TrainConfig>) => {
    onConfigChange({ ...config, ...patch })
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training launch</CardTitle>
          <Button size="sm" variant="ghost" onClick={onReset}>
            Reset config
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-x-3 gap-y-2">
          <label className="col-span-2 text-xs font-medium text-muted-foreground">
            Dataset
          </label>
          {datasetNames.length > 0 ? (
            <>
              <select
                className="col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                aria-label="Training dataset"
                value={customDataset ? '__custom__' : config.dataset}
                onChange={(e) => {
                  if (e.target.value === '__custom__') {
                    setCustomDataset(true)
                  } else {
                    setCustomDataset(false)
                    updateConfig({ dataset: e.target.value })
                  }
                }}
              >
                <option value="__custom__">Custom…</option>
                {[
                  ...datasetNames,
                  ...(datasetNames.includes(config.dataset) ? [] : [config.dataset]),
                ].map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              {customDataset && (
                <>
                  <input
                    className="col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                    value={config.dataset}
                    aria-label="Training dataset"
                    placeholder="Custom dataset name"
                    onChange={(e) => updateConfig({ dataset: e.target.value })}
                  />
                  {config.dataset.trim() !== '' &&
                    !datasetNames.includes(config.dataset.trim()) && (
                      <p className="col-span-2 text-xs text-destructive">
                        Unknown dataset &quot;{config.dataset.trim()}&quot; — Training will fail
                        to start. Available: {datasetNames.slice(0,5).join(', ')}
                        {datasetNames.length > 5 ? ` +${datasetNames.length - 5} more` : ''}.
                      </p>
                    )}
                </>
              )}
            </>
          ) : (
            <input
              className="col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
              value={config.dataset}
              aria-label="Training dataset"
              onChange={(e) => updateConfig({ dataset: e.target.value })}
            />
          )}
          <label className="text-xs font-medium text-muted-foreground">Epochs</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={config.epochs}
            aria-label="Training epochs"
            onChange={(e) => updateConfig({ epochs: num(e.target.value) })}
          />
          <label className="text-xs font-medium text-muted-foreground">Learning rate</label>
          <input
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={config.lr}
            aria-label="Training learning rate"
            onChange={(e) => updateConfig({ lr: num(e.target.value) })}
          />
          <label className="text-xs font-medium text-muted-foreground">Batch size</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={config.batch_size}
            aria-label="Training batch size"
            onChange={(e) => updateConfig({ batch_size: num(e.target.value) })}
          />
          <label className="text-xs font-medium text-muted-foreground">Layers</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={config.n_layer}
            aria-label="Training layers"
            onChange={(e) => updateConfig({ n_layer: num(e.target.value) })}
          />
          <label className="text-xs font-medium text-muted-foreground">Heads</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={config.n_head}
            aria-label="Training heads"
            onChange={(e) => updateConfig({ n_head: num(e.target.value) })}
          />
          <label className="text-xs font-medium text-muted-foreground">Embed size</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={config.embed_dim}
            aria-label="Training embed size"
            onChange={(e) => updateConfig({ embed_dim: num(e.target.value) })}
          />
        </div>
        {hints.length > 0 && (
          <ul className="space-y-0.5 text-xs text-warning">
            {hints.map((hint) => (
              <li key={hint.label}>
                <span className="font-medium">{hint.label}</span>: {hint.message}
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-muted-foreground">
          Generates the <span className="font-mono">train</span> sample with this config and runs
          it. Requires the <span className="font-medium">admin</span> role — the Training card
          polls the job and shows the final result.
        </p>
        {role === 'user' && (
          <div className="flex items-center justify-between gap-2 rounded border border-warning/40 bg-warning/10 px-2 py-1.5">
            <p className="text-xs text-warning">
              Training is denied for the user role (EAX = -2).
            </p>
            <Button size="sm" variant="outline" onClick={() => onRoleChange('admin')}>
              Switch to admin
            </Button>
          </div>
        )}
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={onLoadSample}>
            Load sample
          </Button>
          <Button size="sm" onClick={onLaunch}>
            Launch training
          </Button>
        </div>
        {launchedJob != null && (
          <div className="flex items-center justify-between gap-2 rounded border border-success/40 bg-success/10 px-2 py-1.5">
            <p className="text-xs text-success">
              Launched training job #{launchedJob} — the Training card below polls it to completion.
            </p>
            <button
              type="button"
              className="text-xs underline text-success hover:text-success/80"
              onClick={onDismissJob}
            >
              Dismiss
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
