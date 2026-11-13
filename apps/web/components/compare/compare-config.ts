import type { BenchmarkResult } from '@/lib/benchmark-controller'

export interface MetricColumn {
  key: string
  label: string
  accessor: (r: BenchmarkResult) => number
  fmt: (v: number) => string
  lowerBetter?: boolean
}

export const METRIC_COLUMNS: MetricColumn[] = [
  {
    key: 'throughput_tokens_per_sec',
    label: 'Throughput',
    accessor: r => r.throughput_tokens_per_sec,
    fmt: v => `${v.toFixed(1)} tok/s`,
  },
  {
    key: 'inference_time_ms',
    label: 'Latency',
    lowerBetter: true,
    accessor: r => r.inference_time_ms,
    fmt: v => `${v.toFixed(0)} ms`,
  },
  {
    key: 'latency_p95_ms',
    label: 'P95',
    lowerBetter: true,
    accessor: r => r.latency_p95_ms ?? 0,
    fmt: v => `${v.toFixed(0)} ms`,
  },
  {
    key: 'num_parameters',
    label: 'Params',
    accessor: r => r.num_parameters,
    fmt: v => {
      if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`
      if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`
      return `${v}`
    },
  },
  {
    key: 'memory_mb',
    label: 'Memory',
    lowerBetter: true,
    accessor: r => r.memory_mb,
    fmt: v => `${v.toFixed(0)} MB`,
  },
  {
    key: 'perplexity',
    label: 'PPL',
    lowerBetter: true,
    accessor: r => r.perplexity ?? 0,
    fmt: v => v > 0 ? v.toFixed(3) : '—',
  },
  {
    key: 'bleu',
    label: 'BLEU',
    accessor: r => r.bleu ?? 0,
    fmt: v => v > 0 ? v.toFixed(3) : '—',
  },
]
