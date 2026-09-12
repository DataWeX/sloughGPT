'use client'

import { useState, useCallback, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

type TestType = 'reflect' | 'process' | 'seed' | 'batch' | 'status'

interface BenchmarkResult {
  id: string
  testType: TestType
  iterations: number
  warmup: number
  times: number[]
  avgTime: number
  minTime: number
  maxTime: number
  p50: number
  p95: number
  p99: number
  totalTime: number
  throughput: number
  timestamp: number
}

interface PreviousRun {
  id: string
  testType: TestType
  iterations: number
  avgTime: number
  throughput: number
  timestamp: number
}

const STORAGE_KEY = 'consciousness_benchmark_results'

const TEST_TYPES: { id: TestType; labelKey: string }[] = [
  { id: 'reflect', labelKey: 'consciousness_benchmark.test_reflect' },
  { id: 'process', labelKey: 'consciousness_benchmark.test_process' },
  { id: 'seed', labelKey: 'consciousness_benchmark.test_seed' },
  { id: 'batch', labelKey: 'consciousness_benchmark.test_batch' },
  { id: 'status', labelKey: 'consciousness_benchmark.test_status' },
]

async function apiCall(method: string, path: string, body?: string): Promise<{ status: number; duration: number }> {
  const start = Date.now()
  try {
    const init: RequestInit = {
      method,
      headers: { 'Content-Type': 'application/json' },
    }
    if (body) init.body = body
    await fetch(`${PUBLIC_API_URL}${path}`, init)
    return { status: 200, duration: Date.now() - start }
  } catch {
    return { status: 0, duration: Date.now() - start }
  }
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1
  return sorted[Math.max(0, idx)]
}

function loadPreviousRuns(): PreviousRun[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    return JSON.parse(raw) as PreviousRun[]
  } catch {
    return []
  }
}

function saveRun(result: BenchmarkResult) {
  const runs = loadPreviousRuns()
  const trimmed = runs.slice(-49)
  trimmed.push({
    id: result.id,
    testType: result.testType,
    iterations: result.iterations,
    avgTime: result.avgTime,
    throughput: result.throughput,
    timestamp: result.timestamp,
  })
  localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed))
}

function computeHistogramBins(times: number[], bins: number): { start: number; count: number }[] {
  if (times.length === 0) return []
  const min = Math.min(...times)
  const max = Math.max(...times)
  const range = max - min || 1
  const binSize = range / bins
  const result: { start: number; count: number }[] = []
  for (let i = 0; i < bins; i++) {
    const start = min + i * binSize
    const end = start + binSize
    const count = times.filter(t => i === bins - 1 ? t >= start && t <= end : t >= start && t < end).length
    result.push({ start: Math.round(start), count })
  }
  return result
}

export default function ConsciousnessBenchmarkPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()

  const [iterations, setIterations] = useState(50)
  const [testType, setTestType] = useState<TestType>('reflect')
  const [warmup, setWarmup] = useState(3)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [previousRuns] = useState<PreviousRun[]>(() => loadPreviousRuns())
  const abortRef = useRef(false)

  const runBenchmark = useCallback(async () => {
    if (running) return
    abortRef.current = false
    setRunning(true)
    setProgress(0)
    setResult(null)

    const times: number[] = []
    const total = warmup + iterations

    for (let i = 0; i < total; i++) {
      if (abortRef.current) break

      let duration: number
      switch (testType) {
        case 'reflect': {
          const r = await apiCall('POST', '/consciousness/reflect', '{"text": "benchmark test"}')
          duration = r.duration
          break
        }
        case 'process': {
          const r = await apiCall('POST', '/consciousness/reflect', '{"text": "benchmark process"}')
          duration = r.duration
          break
        }
        case 'seed': {
          const r = await apiCall('POST', '/consciousness/seed?count=1')
          duration = r.duration
          break
        }
        case 'batch': {
          const r = await apiCall('POST', '/consciousness/reflect', '{"text": "benchmark batch"}')
          duration = r.duration
          break
        }
        case 'status': {
          const r = await apiCall('GET', '/consciousness/status')
          duration = r.duration
          break
        }
        default:
          duration = 0
      }

      if (i >= warmup) {
        times.push(duration)
      }
      setProgress(i + 1)
    }

    if (times.length === 0) {
      setRunning(false)
      addToast(t('consciousness_benchmark.toast_aborted'), 'error')
      return
    }

    const sorted = [...times].sort((a, b) => a - b)
    const avgTime = times.reduce((s, v) => s + v, 0) / times.length
    const totalTime = times.reduce((s, v) => s + v, 0)
    const throughput = totalTime > 0 ? (times.length / totalTime) * 1000 : 0

    const benchmarkResult: BenchmarkResult = {
      id: `bench_${Date.now()}`,
      testType,
      iterations: times.length,
      warmup,
      times,
      avgTime: Math.round(avgTime * 100) / 100,
      minTime: sorted[0],
      maxTime: sorted[sorted.length - 1],
      p50: percentile(sorted, 50),
      p95: percentile(sorted, 95),
      p99: percentile(sorted, 99),
      totalTime,
      throughput: Math.round(throughput * 100) / 100,
      timestamp: Date.now(),
    }

    saveRun(benchmarkResult)
    setResult(benchmarkResult)
    setRunning(false)
    addToast(t('consciousness_benchmark.toast_complete'), 'success')
  }, [running, testType, iterations, warmup, addToast, t])

  const stopBenchmark = useCallback(() => {
    abortRef.current = true
  }, [])

  const exportResults = useCallback(() => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `consciousness-benchmark-${result.testType}-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    addToast(t('consciousness_benchmark.toast_exported'), 'success')
  }, [result, addToast, t])

  const progressPct = warmup + iterations > 0 ? (progress / (warmup + iterations)) * 100 : 0

  const prevForType = previousRuns.filter(r => r.testType === testType)
  const lastPrev = prevForType.length > 0 ? prevForType[prevForType.length - 1] : null
  const improvement = result && lastPrev
    ? Math.round(((lastPrev.avgTime - result.avgTime) / lastPrev.avgTime) * 10000) / 100
    : null

  const histBins = result ? computeHistogramBins(result.times, 20) : []
  const maxBinCount = histBins.length > 0 ? Math.max(...histBins.map(b => b.count)) : 1

  return (
    <PageContainer title={t('consciousness_benchmark.page_title')}>
      <div className="space-y-6 p-6">

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_benchmark.config_title')}</CardTitle>
            <CardDescription>{t('consciousness_benchmark.config_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-1.5">
                <label className="text-sm text-muted-foreground">{t('consciousness_benchmark.iterations_label')}</label>
                <input
                  type="number"
                  min={10}
                  max={1000}
                  step={10}
                  value={iterations}
                  onChange={e => setIterations(Math.max(10, Math.min(1000, Number(e.target.value))))}
                  disabled={running}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm font-mono focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-sm text-muted-foreground">{t('consciousness_benchmark.test_type_label')}</label>
                <select
                  value={testType}
                  onChange={e => setTestType(e.target.value as TestType)}
                  disabled={running}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
                >
                  {TEST_TYPES.map(tt => (
                    <option key={tt.id} value={tt.id}>{t(tt.labelKey)}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-sm text-muted-foreground">{t('consciousness_benchmark.warmup_label')}</label>
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={warmup}
                  onChange={e => setWarmup(Math.max(0, Math.min(10, Number(e.target.value))))}
                  disabled={running}
                  className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm font-mono focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={runBenchmark} disabled={running}>
                {t('consciousness_benchmark.run_benchmark')}
              </Button>
              {running && (
                <Button onClick={stopBenchmark} variant="destructive">
                  {t('consciousness_benchmark.stop')}
                </Button>
              )}
              <div className="flex-1" />
              <div className="text-xs text-muted-foreground">
                {t('consciousness_benchmark.progress', { current: Math.max(0, progress - warmup), total: iterations })}
              </div>
            </div>
            {running && (
              <div className="w-full h-2 bg-muted/30 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            )}
          </CardContent>
        </Card>

        {result && (
          <>
            <Card>
              <CardHeader>
                <CardTitle>{t('consciousness_benchmark.summary_title')}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground text-xs">{t('consciousness_benchmark.avg_time')}</div>
                    <div className="font-mono text-lg">{result.avgTime}ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-xs">{t('consciousness_benchmark.min_time')}</div>
                    <div className="font-mono text-lg">{result.minTime}ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-xs">{t('consciousness_benchmark.max_time')}</div>
                    <div className="font-mono text-lg">{result.maxTime}ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-xs">{t('consciousness_benchmark.total_time')}</div>
                    <div className="font-mono text-lg">{result.totalTime}ms</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground text-xs">P50</div>
                    <div className="font-mono text-lg">{result.p50}ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-xs">P95</div>
                    <div className="font-mono text-lg">{result.p95}ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-xs">P99</div>
                    <div className="font-mono text-lg">{result.p99}ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-xs">{t('consciousness_benchmark.throughput')}</div>
                    <div className="font-mono text-lg">{result.throughput} req/s</div>
                  </div>
                </div>
                {improvement !== null && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">{t('consciousness_benchmark.vs_previous')}:</span>
                    <Badge className={improvement >= 0 ? 'bg-green-500/15 text-green-400 border-green-500/30' : 'bg-red-500/15 text-red-400 border-red-500/30'}>
                      {improvement >= 0 ? `+${improvement}%` : `${improvement}%`}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {t(improvement >= 0 ? 'consciousness_benchmark.improvement' : 'consciousness_benchmark.regression')}
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{t('consciousness_benchmark.histogram_title')}</CardTitle>
                <CardDescription>{t('consciousness_benchmark.histogram_desc')}</CardDescription>
              </CardHeader>
              <CardContent>
                <svg viewBox="0 0 800 200" className="w-full h-48" role="img" aria-label="Latency distribution histogram">
                  {histBins.map((bin, i) => {
                    const barWidth = 800 / histBins.length - 2
                    const barHeight = maxBinCount > 0 ? (bin.count / maxBinCount) * 180 : 0
                    const x = i * (800 / histBins.length) + 1
                    const y = 190 - barHeight
                    return (
                      <g key={i}>
                        <rect
                          x={x}
                          y={y}
                          width={barWidth}
                          height={barHeight}
                          fill="rgb(var(--primary))"
                          opacity={0.7}
                          rx={2}
                        />
                        {i % Math.max(1, Math.floor(histBins.length / 8)) === 0 && (
                          <text
                            x={x + barWidth / 2}
                            y={198}
                            textAnchor="middle"
                            className="fill-muted-foreground"
                            fontSize={10}
                          >
                            {bin.start}
                          </text>
                        )}
                      </g>
                    )
                  })}
                </svg>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{t('consciousness_benchmark.timeline_title')}</CardTitle>
                <CardDescription>{t('consciousness_benchmark.timeline_desc')}</CardDescription>
              </CardHeader>
              <CardContent>
                <svg viewBox="0 0 800 200" className="w-full h-48" role="img" aria-label="Response time timeline">
                  {result.times.length > 1 && (() => {
                    const maxVal = Math.max(...result.times)
                    const minVal = Math.min(...result.times)
                    const range = maxVal - minVal || 1
                    const points = result.times.map((t, i) => {
                      const x = (i / (result.times.length - 1)) * 780 + 10
                      const y = 190 - ((t - minVal) / range) * 170
                      return `${x},${y}`
                    }).join(' ')
                    return (
                      <>
                        <polyline
                          points={points}
                          fill="none"
                          stroke="rgb(var(--primary))"
                          strokeWidth={1.5}
                          opacity={0.8}
                        />
                        <line x1={10} y1={190} x2={790} y2={190} stroke="rgb(var(--border))" strokeWidth={0.5} />
                        <text x={5} y={12} className="fill-muted-foreground" fontSize={10}>{maxVal}ms</text>
                        <text x={5} y={188} className="fill-muted-foreground" fontSize={10}>{minVal}ms</text>
                      </>
                    )
                  })()}
                </svg>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{t('consciousness_benchmark.comparison_title')}</CardTitle>
                <CardDescription>{t('consciousness_benchmark.comparison_desc')}</CardDescription>
              </CardHeader>
              <CardContent>
                {prevForType.length === 0 ? (
                  <div className="text-sm text-muted-foreground">{t('consciousness_benchmark.no_previous')}</div>
                ) : (
                  <div className="rounded-md border border-border/50 overflow-hidden">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-muted/30">
                          <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_benchmark.col_date')}</th>
                          <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_benchmark.col_iterations')}</th>
                          <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_benchmark.col_avg_time')}</th>
                          <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_benchmark.col_throughput')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {prevForType.slice(-10).reverse().map(run => (
                          <tr key={run.id} className="border-t border-border/30">
                            <td className="px-3 py-1.5 font-mono">{new Date(run.timestamp).toLocaleString()}</td>
                            <td className="px-3 py-1.5 font-mono">{run.iterations}</td>
                            <td className="px-3 py-1.5 font-mono">{run.avgTime}ms</td>
                            <td className="px-3 py-1.5 font-mono">{run.throughput} req/s</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>

            <div className="flex gap-2">
              <Button variant="outline" onClick={exportResults}>
                {t('consciousness_benchmark.export_json')}
              </Button>
            </div>
          </>
        )}

      </div>
    </PageContainer>
  )
}
