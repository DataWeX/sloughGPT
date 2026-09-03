import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent, within } from '@testing-library/react'
import { SystemHealthPanel } from './SystemHealthPanel'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

vi.mock('@/hooks/useTick', () => ({ useTick: vi.fn() }))
vi.mock('@/lib/chat-utils', () => ({ formatUptime: (s: number) => `${Math.floor(s / 3600)}h` }))
vi.mock('@/lib/time-ago', () => ({ timeAgo: () => '2m ago' }))

function makeLiveHealth(overrides: Partial<LiveHealthSnapshot> = {}): LiveHealthSnapshot {
  return {
    model_loaded: false, model_loading: false, model_type: null, device: null, soul: null,
    is_inferencing: false, inference_count: 0, uptime_seconds: 0, request_count: 0,
    error_count: 0, tokens_per_sec: 0, avg_latency_ms: 0, p95_latency_ms: 0,
    requests_per_minute: 0, total_tokens: 0, avg_tokens_per_request: 0,
    cpu_percent: null, memory_percent: null, health_score: 0, health_status: 'unknown',
    health_summary: '', diagnoses: [], num_parameters: null, quantization: null,
    training_pool: null, model_metrics: [], model_events: [], rate_violations: [],
    health_history: [], memory_history: [], path_latencies: [], recent_errors: [],
    ...overrides,
  }
}

const emptyProps = {
  liveHealth: null, detailed: null, metrics: null, disk: null, info: null,
  connectionStatus: 'connected' as const, loaded: true, chartHistory: [],
}

describe('SystemHealthPanel', () => {
  describe('loading state', () => {
    it('renders skeletons when not loaded', () => {
      const { container } = render(<SystemHealthPanel {...emptyProps} loaded={false} />)
      expect(container.querySelectorAll('[class*="animate-pulse"]').length).toBeGreaterThanOrEqual(1)
    })
  })

  describe('connection status', () => {
    it('shows Connected', () => {
      render(<SystemHealthPanel {...emptyProps} />)
      expect(screen.getByText('Connected')).toBeDefined()
    })
    it('shows Reconnecting', () => {
      render(<SystemHealthPanel {...emptyProps} connectionStatus="connecting" />)
      expect(screen.getByText('Reconnecting')).toBeDefined()
    })
    it('shows Offline', () => {
      render(<SystemHealthPanel {...emptyProps} connectionStatus="offline" />)
      expect(screen.getByText('Offline')).toBeDefined()
    })
  })

  describe('health ring', () => {
    it('shows health score', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ health_score: 85, health_status: 'good' })} />)
      expect(screen.getByText('85')).toBeDefined()
    })
  })

  describe('resources', () => {
    it('shows CPU bar', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ cpu_percent: 45.5 })} />)
      expect(screen.getByText('45.5%')).toBeDefined()
      expect(screen.getByText('CPU')).toBeDefined()
    })
    it('shows memory bar', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ memory_percent: 72.3 })} />)
      expect(screen.getByText('72.3%')).toBeDefined()
    })
  })

  describe('model info', () => {
    it('shows Loaded', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ model_loaded: true })} />)
      expect(screen.getByText('Loaded')).toBeDefined()
    })
    it('shows Loading...', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ model_loading: true })} />)
      expect(screen.getByText('Loading...')).toBeDefined()
    })
    it('shows model type', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ model_type: 'gpt2' })} />)
      expect(screen.getByText('gpt2')).toBeDefined()
    })
    it('shows parameters in B format', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ num_parameters: 1_500_000_000 })} />)
      expect(screen.getByText('1.5B')).toBeDefined()
    })
    it('shows parameters in M format', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ num_parameters: 124_000_000 })} />)
      expect(screen.getByText('124M')).toBeDefined()
    })
    it('shows device', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ device: 'cuda:0' })} />)
      expect(screen.getByText('cuda:0')).toBeDefined()
    })
    it('shows soul', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ soul: 'empathetic' })} />)
      expect(screen.getByText('empathetic')).toBeDefined()
    })
  })

  describe('inference stats', () => {
    it('shows request count', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ request_count: 42 })} />)
      expect(screen.getByText('42')).toBeDefined()
    })
    it('shows tokens/sec', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ tokens_per_sec: 15.3 })} />)
      expect(screen.getByText('15.3')).toBeDefined()
    })
    it('shows total tokens in K format', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ total_tokens: 5000 })} />)
      expect(screen.getByText('5.0K')).toBeDefined()
    })
    it('shows total tokens in M format', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ total_tokens: 2_000_000 })} />)
      expect(screen.getByText('2.0M')).toBeDefined()
    })
  })

  describe('uptime', () => {
    it('shows uptime from liveHealth', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ uptime_seconds: 7200 })} />)
      expect(screen.getByText(/2h/)).toBeDefined()
    })
    it('shows uptime from detailed', () => {
      render(<SystemHealthPanel {...emptyProps} detailed={{ uptime_seconds: 3600 } as any} />)
      expect(screen.getByText(/1h/)).toBeDefined()
    })
  })

  describe('expandable sections', () => {
    it('shows path latencies expandable', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ path_latencies: [{ path: '/api/chat', count: 10, avg_ms: 5.2, p95_ms: 12.1 }] })} />)
      expect(screen.getByText('Endpoint Latency')).toBeDefined()
    })

    it('shows recent errors expandable', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ recent_errors: [{ ts: 1, status: 500, method: 'GET', path: '/api', error_type: 'Error', message: 'fail' }] })} />)
      expect(screen.getByText('Recent Errors')).toBeDefined()
    })
    it('shows rate violations expandable', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ rate_violations: [{ path: '/api', count: 5, limit: 10, ts: 1 }] })} />)
      expect(screen.getByText('Rate Violations')).toBeDefined()
    })
    it('shows model metrics expandable', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ model_metrics: [{ model: 'gpt2', count: 100, total_tokens: 5000, tokens_per_sec: 15, avg_tokens: 50 }] })} />)
      expect(screen.getByText('Model Metrics')).toBeDefined()
    })
  })

  describe('disk info', () => {
    it('shows disk usage', () => {
      render(<SystemHealthPanel {...emptyProps} disk={{ used_gb: 50, total_gb: 100, free_gb: 50, percent: 50 } as any} />)
      expect(screen.getByText(/50 \/ 100 GB/)).toBeDefined()
    })
  })

  describe('system info', () => {
    it('shows platform', () => {
      render(<SystemHealthPanel {...emptyProps} info={{ platform: 'linux', platform_release: '6.1', architecture: 'x86_64', cpu_count: 8 } as any} />)
      expect(screen.getByText(/linux/)).toBeDefined()
    })
  })

  describe('diagnoses', () => {
    it('shows diagnoses strip', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ diagnoses: [{ check: 'memory', severity: 'warn', score: 0, message: 'high usage' }] })} />)
      expect(screen.getByText('memory')).toBeDefined()
      expect(screen.getByText('high usage')).toBeDefined()
    })
  })

  describe('sparklines', () => {
    it('renders sparklines with enough data', () => {
      const { container } = render(<SystemHealthPanel {...emptyProps} chartHistory={[{ time: '1', cpu: 10, mem: 20 }, { time: '2', cpu: 15, mem: 25 }]} />)
      expect(container.querySelectorAll('svg').length).toBeGreaterThanOrEqual(1)
    })
  })

  describe('lifecycle badge', () => {
    it('shows lifecycle phase', () => {
      render(<SystemHealthPanel {...emptyProps} detailed={{ lifecycle: { phase: 'running', is_running: true, profile: 'production' }, uptime_seconds: 0 } as any} />)
      expect(screen.getByText('running')).toBeDefined()
    })
  })

  describe('process guard badge', () => {
    it('shows guard active', () => {
      render(<SystemHealthPanel {...emptyProps} detailed={{ process_guard: { active: true, enabled: true }, uptime_seconds: 0 } as any} />)
      expect(screen.getByText('guard active')).toBeDefined()
    })
  })

  describe('memory pressure', () => {
    it('shows current and peak RSS', () => {
      render(<SystemHealthPanel {...emptyProps} detailed={{ memory_pressure: { current_mb: 512, peak_mb: 1024, pressure_level: 'moderate', tracked_count: 10 }, uptime_seconds: 0 } as any} />)
      expect(screen.getByText('512 MB')).toBeDefined()
      expect(screen.getByText('1024 MB')).toBeDefined()
    })
  })

  describe('kv sessions', () => {
    it('shows kv sessions when enabled', () => {
      render(<SystemHealthPanel {...emptyProps} detailed={{ kv_sessions: { enabled: true, active_sessions: 3, max_sessions: 10, cached_tokens: 5000 }, uptime_seconds: 0 } as any} />)
      expect(screen.getByText('KV sessions')).toBeDefined()
    })
  })

  describe('resource allocation', () => {
    it('shows expandable with thread info', () => {
      render(<SystemHealthPanel {...emptyProps} detailed={{ resource_allocation: { mode: 'balanced', compute_threads: 4, io_threads: 2 }, uptime_seconds: 0 } as any} />)
      expect(screen.getByText('Resource Allocation')).toBeDefined()
      fireEvent.click(screen.getByText('Resource Allocation'))
      expect(screen.getByText('balanced')).toBeDefined()
    })
  })

  describe('inferencing badge', () => {
    it('shows inferencing when active', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ is_inferencing: true })} />)
      expect(screen.getByText('inferencing')).toBeDefined()
    })
  })

  describe('model events', () => {
    it('shows model events strip', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ model_events: [{ type: 'load', model: 'gpt2', detail: 'loaded', ts: 1 }] })} />)
      expect(screen.getByText('load')).toBeDefined()
    })
  })

  describe('health summary', () => {
    it('shows health summary', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={makeLiveHealth({ health_summary: 'System is healthy' })} />)
      expect(screen.getByText('System is healthy')).toBeDefined()
    })
  })
})
