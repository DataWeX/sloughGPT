import { describe, it, expect } from 'vitest'

import { METRIC_COLUMNS } from './compare-config'
import type { BenchmarkResult } from '@/lib/benchmark-controller'

const sample: BenchmarkResult = {
  model: 'gpt2',
  num_parameters: 124000000,
  memory_mb: 500,
  throughput_tokens_per_sec: 12.345,
  inference_time_ms: 123.456,
  latency_p95_ms: 200,
  latency_p99_ms: 300,
  latency_ms: 123,
  throughput: 12,
}

describe('compare-config METRIC_COLUMNS', () => {
  it('defines 6 metric columns', () => {
    expect(METRIC_COLUMNS).toHaveLength(6)
  })

  it('every column has label, key, fmt, accessor; lowerBetter is optional', () => {
    for (const col of METRIC_COLUMNS) {
      expect(typeof col.label).toBe('string')
      expect(typeof col.key).toBe('string')
      expect(typeof col.fmt).toBe('function')
      expect(typeof col.accessor).toBe('function')
      expect(col.lowerBetter === undefined || typeof col.lowerBetter === 'boolean').toBe(true)
    }
  })

  it('column keys exist on BenchmarkResult', () => {
    const keys = METRIC_COLUMNS.map(c => c.key)
    for (const key of keys) {
      expect(key in sample).toBe(true)
    }
  })

  it('formats each metric', () => {
    const byLabel = Object.fromEntries(METRIC_COLUMNS.map(c => [c.label, c]))
    expect(byLabel['Model size'].fmt(124000000)).toBe('124,000,000')
    expect(byLabel['Memory'].fmt(500)).toBe('500 MB')
    expect(byLabel['Speed'].fmt(12.345)).toBe('12.3 tok/s')
    expect(byLabel['Avg response time'].fmt(123.456)).toBe('123 ms')
    expect(byLabel['95th percentile'].fmt(200)).toBe('200 ms')
    expect(byLabel['99th percentile'].fmt(300)).toBe('300 ms')
  })

  it('accessors read the right fields', () => {
    const byKey = Object.fromEntries(METRIC_COLUMNS.map(c => [c.key, c]))
    expect(byKey.num_parameters.accessor(sample)).toBe(124000000)
    expect(byKey.memory_mb.accessor(sample)).toBe(500)
    expect(byKey.throughput_tokens_per_sec.accessor(sample)).toBeCloseTo(12.345)
    expect(byKey.inference_time_ms.accessor(sample)).toBeCloseTo(123.456)
    expect(byKey.latency_p95_ms.accessor(sample)).toBe(200)
    expect(byKey.latency_p99_ms.accessor(sample)).toBe(300)
  })

  it('lowerBetter marks the latency columns only', () => {
    const lower = METRIC_COLUMNS.filter(c => c.lowerBetter).map(c => c.label)
    expect(lower).toEqual(['Avg response time', '95th percentile', '99th percentile'])
  })

  it('accessors fall back to Infinity when p95/p99 are missing', () => {
    const missing = { ...sample, latency_p95_ms: undefined, latency_p99_ms: undefined }
    const p95 = METRIC_COLUMNS.find(c => c.key === 'latency_p95_ms')!
    const p99 = METRIC_COLUMNS.find(c => c.key === 'latency_p99_ms')!
    expect(p95.accessor(missing)).toBe(Infinity)
    expect(p99.accessor(missing)).toBe(Infinity)
    expect(p95.fmt(p95.accessor(missing))).toBe('Infinity ms')
  })
})
