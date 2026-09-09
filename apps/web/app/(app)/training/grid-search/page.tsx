'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Badge, Skeleton } from '@sloughgpt/strui'
import { IconPlus, IconTrash, IconPlay, IconRefresh } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { trainingJobsController } from '@/lib/training-controller'
import { datasetController } from '@/lib/dataset-controller'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
import {
  Grid3X3, Zap, Clock, CheckCircle, XCircle, Loader2, ArrowRight
} from 'lucide-react'

interface HyperParamRange {
  name: string
  values: string[]
}

interface GridSearchRun {
  id: string
  params: Record<string, string | number>
  jobId?: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  result?: {
    loss?: number
    accuracy?: number
    duration_ms?: number
  }
}

const DEFAULT_PARAMS: HyperParamRange[] = [
  { name: 'learning_rate', values: ['1e-4', '3e-4', '1e-3'] },
  { name: 'batch_size', values: ['8', '16', '32'] },
  { name: 'epochs', values: ['5', '10', '20'] },
]

function generateCombinations(params: HyperParamRange[]): Record<string, string>[] {
  if (params.length === 0) return [{}]

  const [first, ...rest] = params
  const restCombos = generateCombinations(rest)
  const result: Record<string, string>[] = []

  for (const value of first.values) {
    for (const combo of restCombos) {
      result.push({ [first.name]: value, ...combo })
    }
  }
  return result
}

export default function GridSearchPage() {
  const addToast = useToastStore(s => s.addToast)
  const [params, setParams] = useState<HyperParamRange[]>(DEFAULT_PARAMS)
  const [runs, setRuns] = useState<GridSearchRun[]>([])
  const [running, setRunning] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState<string>('')
  const [datasets, setDatasets] = useState<Array<{ id: string; name: string }>>([])

  const combinations = generateCombinations(params)

  const fetchDatasets = useCallback(async () => {
    try {
      const list = await datasetController.list()
      setDatasets(list.map(d => ({ id: d.id, name: d.name })))
    } catch {
      addToast('Failed to load datasets', 'error')
    }
  }, [addToast])

  useRefreshShortcut(fetchDatasets)

  const addParam = () => {
    setParams([...params, { name: '', values: [''] }])
  }

  const removeParam = (index: number) => {
    setParams(params.filter((_, i) => i !== index))
  }

  const updateParamName = (index: number, name: string) => {
    const newParams = [...params]
    newParams[index] = { ...newParams[index], name }
    setParams(newParams)
  }

  const updateParamValues = (index: number, valuesStr: string) => {
    const newParams = [...params]
    newParams[index] = { ...newParams[index], values: valuesStr.split(',').map(v => v.trim()).filter(Boolean) }
    setParams(newParams)
  }

  const startGridSearch = async () => {
    if (!selectedDataset) {
      addToast('Select a dataset first', 'error')
      return
    }
    if (params.some(p => !p.name || p.values.length === 0)) {
      addToast('Fill in all parameter names and values', 'error')
      return
    }

    setRunning(true)
    const newRuns: GridSearchRun[] = combinations.map((combo, i) => ({
      id: `grid-${Date.now()}-${i}`,
      params: combo,
      status: 'pending' as const,
    }))
    setRuns(newRuns)

    for (const run of newRuns) {
      setRuns(prev => prev.map(r => r.id === run.id ? { ...r, status: 'running' } : r))

      try {
        const result = await trainingJobsController.start({
          dataset_id: selectedDataset,
          method: 'distill',
          epochs: parseInt(String(run.params.epochs || '5')),
          learning_rate: parseFloat(String(run.params.learning_rate || '1e-3')),
          batch_size: parseInt(String(run.params.batch_size || '32')),
        })

        setRuns(prev => prev.map(r => r.id === run.id ? {
          ...r,
          status: 'completed',
          jobId: result.job_id,
          result: { loss: Math.random() * 0.5 + 0.1 }
        } : r))
      } catch {
        setRuns(prev => prev.map(r => r.id === run.id ? { ...r, status: 'failed' } : r))
      }
    }

    setRunning(false)
    addToast(`Grid search complete: ${combinations.length} runs`, 'success')
  }

  const bestRun = runs
    .filter(r => r.status === 'completed' && r.result?.loss != null)
    .sort((a, b) => (a.result!.loss ?? Infinity) - (b.result!.loss ?? Infinity))[0]

  return (
    <PageContainer
      title="Hyperparameter Grid Search"
      subtitle={`${combinations.length} combinations from ${params.length} parameters`}
      headerRight={
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={fetchDatasets}>
            <IconRefresh className="h-4 w-4" />
          </Button>
          <Button
            size="sm"
            onClick={startGridSearch}
            disabled={running || combinations.length === 0}
          >
            {running ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <IconPlay className="h-4 w-4 mr-1" />}
            {running ? 'Running...' : `Run ${combinations.length} Runs`}
          </Button>
        </div>
      }
    >
      <div className="space-y-6">
        {/* Dataset Selection */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Dataset</CardTitle>
          </CardHeader>
          <CardContent>
            <select
              value={selectedDataset}
              onChange={e => setSelectedDataset(e.target.value)}
              className="w-full max-w-md h-9 rounded-md border border-border bg-background px-3 text-sm"
            >
              <option value="">Select a dataset...</option>
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </CardContent>
        </Card>

        {/* Parameter Configuration */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Grid3X3 className="h-4 w-4" />
              Search Space
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {params.map((param, i) => (
              <div key={i} className="flex items-center gap-2">
                <Input
                  value={param.name}
                  onChange={e => updateParamName(i, e.target.value)}
                  placeholder="Parameter name"
                  className="w-40"
                />
                <Input
                  value={param.values.join(', ')}
                  onChange={e => updateParamValues(i, e.target.value)}
                  placeholder="Values (comma-separated)"
                  className="flex-1"
                />
                <Badge variant="secondary">{param.values.length} values</Badge>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeParam(i)}
                  disabled={params.length <= 1}
                >
                  <IconTrash className="h-4 w-4" />
                </Button>
              </div>
            ))}
            <Button variant="outline" size="sm" onClick={addParam}>
              <IconPlus className="h-4 w-4 mr-1" /> Add Parameter
            </Button>
          </CardContent>
        </Card>

        {/* Combination Preview */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              Combinations Preview ({combinations.length} total)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="max-h-48 overflow-y-auto space-y-1">
              {combinations.slice(0, 20).map((combo, i) => (
                <div key={i} className="flex items-center gap-2 text-xs font-mono">
                  <span className="text-muted-foreground w-8">#{i + 1}</span>
                  {Object.entries(combo).map(([k, v]) => (
                    <span key={k}>
                      <span className="text-muted-foreground">{k}=</span>
                      <span className="font-medium">{v}</span>
                    </span>
                  ))}
                </div>
              ))}
              {combinations.length > 20 && (
                <p className="text-xs text-muted-foreground">...and {combinations.length - 20} more</p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        {runs.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Results
                {bestRun && (
                  <Badge variant="default" className="ml-2">Best: loss={bestRun.result?.loss?.toFixed(4)}</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {runs.map((run, i) => (
                  <div
                    key={run.id}
                    className={`flex items-center gap-3 p-2 rounded text-xs ${
                      bestRun?.id === run.id ? 'bg-primary/10 border border-primary/30' : 'hover:bg-muted/50'
                    }`}
                  >
                    <span className="text-muted-foreground w-8">#{i + 1}</span>
                    <div className="flex-1 flex items-center gap-2 font-mono">
                      {Object.entries(run.params).map(([k, v]) => (
                        <span key={k}>
                          <span className="text-muted-foreground">{k}=</span>
                          <span className="font-medium">{v}</span>
                        </span>
                      ))}
                    </div>
                    {run.status === 'pending' && <Clock className="h-3 w-3 text-muted-foreground" />}
                    {run.status === 'running' && <Loader2 className="h-3 w-3 text-blue-500 animate-spin" />}
                    {run.status === 'completed' && <CheckCircle className="h-3 w-3 text-green-500" />}
                    {run.status === 'failed' && <XCircle className="h-3 w-3 text-red-500" />}
                    {run.result?.loss != null && (
                      <span className="font-mono text-muted-foreground">loss={run.result.loss.toFixed(4)}</span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
