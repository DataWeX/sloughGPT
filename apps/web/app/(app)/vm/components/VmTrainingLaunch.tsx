'use client'

// Training launch card: dataset/optimizer config form, config hints, and the
// launch action. Extracted from page.tsx by the page-decomposition refactor —
// pure presentation over hook state.

import { type Dispatch, type SetStateAction } from 'react'
import { Button, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import {
  DEFAULT_TRAIN_CONFIG,
  buildTrainSource,
  clampTrainConfig,
  num,
  trainConfigHints,
  type TrainConfig,
} from '../vm-data'

interface VmTrainingLaunchProps {
  trainConfig: TrainConfig
  setTrainConfig: Dispatch<SetStateAction<TrainConfig>>
  customDataset: boolean
  setCustomDataset: Dispatch<SetStateAction<boolean>>
  datasetNames: string[]
  role: string
  setRole: Dispatch<SetStateAction<string>>
  launchedJob: number | null
  setLaunchedJob: Dispatch<SetStateAction<number | null>>
  setSource: Dispatch<SetStateAction<string>>
  onLaunch: () => void
}

export function VmTrainingLaunch({
  trainConfig,
  setTrainConfig,
  customDataset,
  setCustomDataset,
  datasetNames,
  role,
  setRole,
  launchedJob,
  setLaunchedJob,
  setSource,
  onLaunch,
}: VmTrainingLaunchProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training launch</CardTitle>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setTrainConfig(DEFAULT_TRAIN_CONFIG)
              setCustomDataset(false)
            }}
          >
            Reset config
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-x-3 gap-y-2">
          <label className="col-span-2 text-xs font-medium text-muted-foreground">Dataset</label>
          {datasetNames.length > 0 ? (
            <>
              <select
                className="col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                aria-label="Training dataset"
                value={customDataset ? '__custom__' : trainConfig.dataset}
                onChange={(e) => {
                  if (e.target.value === '__custom__') {
                    setCustomDataset(true)
                  } else {
                    setCustomDataset(false)
                    setTrainConfig((c) => ({
                      ...c,
                      dataset: e.target.value,
                    }))
                  }
                }}
              >
                <option value="__custom__">Custom…</option>
                {[
                  ...datasetNames,
                  ...(datasetNames.includes(trainConfig.dataset)
                    ? []
                    : [trainConfig.dataset]),
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
                    value={trainConfig.dataset}
                    aria-label="Training dataset"
                    placeholder="Custom dataset name"
                    onChange={(e) =>
                      setTrainConfig((c) => ({
                        ...c,
                        dataset: e.target.value,
                      }))
                    }
                  />
                  {trainConfig.dataset.trim() !== '' &&
                    !datasetNames.includes(trainConfig.dataset.trim()) && (
                      <p className="col-span-2 text-xs text-destructive">
                        Unknown dataset &quot;{trainConfig.dataset.trim()}&quot; — Training will
                        fail to start. Available: {datasetNames.slice(0, 5).join(', ')}
                        {datasetNames.length > 5 ? ` +${datasetNames.length - 5} more` : ''}.
                      </p>
                    )}
                </>
              )}
            </>
          ) : (
            <input
              className="col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
              value={trainConfig.dataset}
              aria-label="Training dataset"
              onChange={(e) =>
                setTrainConfig((c) => ({
                  ...c,
                  dataset: e.target.value,
                }))
              }
            />
          )}
          <label className="text-xs font-medium text-muted-foreground">Epochs</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={trainConfig.epochs}
            aria-label="Training epochs"
            onChange={(e) =>
              setTrainConfig((c) => ({
                ...c,
                epochs: num(e.target.value),
              }))
            }
          />
          <label className="text-xs font-medium text-muted-foreground">Learning rate</label>
          <input
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={trainConfig.lr}
            aria-label="Training learning rate"
            onChange={(e) =>
              setTrainConfig((c) => ({
                ...c,
                lr: num(e.target.value),
              }))
            }
          />
          <label className="text-xs font-medium text-muted-foreground">Batch size</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={trainConfig.batch_size}
            aria-label="Training batch size"
            onChange={(e) =>
              setTrainConfig((c) => ({
                ...c,
                batch_size: num(e.target.value),
              }))
            }
          />
          <label className="text-xs font-medium text-muted-foreground">Layers</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={trainConfig.n_layer}
            aria-label="Training layers"
            onChange={(e) =>
              setTrainConfig((c) => ({
                ...c,
                n_layer: num(e.target.value),
              }))
            }
          />
          <label className="text-xs font-medium text-muted-foreground">Heads</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={trainConfig.n_head}
            aria-label="Training heads"
            onChange={(e) =>
              setTrainConfig((c) => ({
                ...c,
                n_head: num(e.target.value),
              }))
            }
          />
          <label className="text-xs font-medium text-muted-foreground">Embed size</label>
          <input
            type="number"
            min={1}
            className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
            value={trainConfig.embed_dim}
            aria-label="Training embed size"
            onChange={(e) =>
              setTrainConfig((c) => ({
                ...c,
                embed_dim: num(e.target.value),
              }))
            }
          />
        </div>
        {trainConfigHints(trainConfig).length > 0 && (
          <ul className="space-y-0.5 text-xs text-warning">
            {trainConfigHints(trainConfig).map((hint) => (
              <li key={hint.label}>
                <span className="font-medium">{hint.label}</span>: {hint.message}
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-muted-foreground">
          Generates the <span className="font-mono">train</span> sample with this config and runs
          it. Requires the <span className="font-medium">admin</span> role — the Training card polls
          the job and shows the final result.
        </p>
        {role === 'user' && (
          <div className="flex items-center justify-between gap-2 rounded border border-warning/40 bg-warning/10 px-2 py-1.5">
            <p className="text-xs text-warning">Training is denied for the user role (EAX = -2).</p>
            <Button size="sm" variant="outline" onClick={() => setRole('admin')}>
              Switch to admin
            </Button>
          </div>
        )}
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setSource(buildTrainSource(clampTrainConfig(trainConfig)))}
          >
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
              onClick={() => setLaunchedJob(null)}
            >
              Dismiss
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
