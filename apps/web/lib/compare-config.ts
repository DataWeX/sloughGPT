'use client'

import type { BenchmarkResult } from '@/lib/benchmark-controller'

export interface CompareMetricColumn {
  label: string
  key: keyof BenchmarkResult
  fmt: (v: number) => string
  lowerBetter?: boolean
  accessor: (r: BenchmarkResult) => number
}

export const METRIC_COLUMNS: CompareMetricColumn[] = [
  { label: 'Model size', key: 'num_parameters', fmt: v => v.toLocaleString(), accessor: r => r.num_parameters },
  { label: 'Memory', key: 'memory_mb', fmt: v => `${v.toFixed(0)} MB`, accessor: r => r.memory_mb },
  { label: 'Speed', key: 'throughput_tokens_per_sec', fmt: v => `${v.toFixed(1)} tok/s`, accessor: r => r.throughput_tokens_per_sec },
  { label: 'Avg response time', key: 'inference_time_ms', fmt: v => `${v.toFixed(0)} ms`, lowerBetter: true, accessor: r => r.inference_time_ms },
  { label: '95th percentile', key: 'latency_p95_ms', fmt: v => `${v.toFixed(0)} ms`, lowerBetter: true, accessor: r => r.latency_p95_ms ?? Infinity },
  { label: '99th percentile', key: 'latency_p99_ms', fmt: v => `${v.toFixed(0)} ms`, lowerBetter: true, accessor: r => r.latency_p99_ms ?? Infinity },
]
