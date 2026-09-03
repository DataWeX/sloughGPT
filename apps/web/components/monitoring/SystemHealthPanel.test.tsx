import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { SystemHealthPanel } from './SystemHealthPanel'

vi.mock('@/hooks/useTick', () => ({ useTick: vi.fn() }))
vi.mock('@/lib/chat-utils', () => ({ formatUptime: (s: number) => `${Math.floor(s / 3600)}h` }))
vi.mock('@/lib/time-ago', () => ({ timeAgo: () => '2m ago' }))

const emptyProps = {
  liveHealth: null,
  detailed: null,
  metrics: null,
  disk: null,
  info: null,
  connectionStatus: 'connected' as const,
  loaded: true,
  chartHistory: [],
}

describe('SystemHealthPanel', () => {
  describe('loading state', () => {
    it('renders skeletons when not loaded', () => {
      const { container } = render(<SystemHealthPanel {...emptyProps} loaded={false} />)
      const skeletons = container.querySelectorAll('[class*="animate-pulse"]')
      expect(skeletons.length).toBeGreaterThanOrEqual(1)
    })
  })

  describe('connection status', () => {
    it('shows Connected when connected', () => {
      render(<SystemHealthPanel {...emptyProps} />)
      expect(screen.getByText('Connected')).toBeDefined()
    })
    it('shows Reconnecting when connecting', () => {
      render(<SystemHealthPanel {...emptyProps} connectionStatus="connecting" />)
      expect(screen.getByText('Reconnecting')).toBeDefined()
    })
    it('shows Offline when disconnected', () => {
      render(<SystemHealthPanel {...emptyProps} connectionStatus="disconnected" />)
      expect(screen.getByText('Offline')).toBeDefined()
    })
  })

  describe('health ring', () => {
    it('shows health score', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ health_score: 85, health_status: 'good' } as any} />)
      expect(screen.getByText('85')).toBeDefined()
    })
    it('shows health status label', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ health_score: 85, health_status: 'good' } as any} />)
      expect(screen.getByText('good')).toBeDefined()
    })
  })

  describe('resources', () => {
    it('shows CPU bar when cpu available', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ cpu_percent: 45.5 } as any} />)
      expect(screen.getByText('45.5%')).toBeDefined()
      expect(screen.getByText('CPU')).toBeDefined()
    })
    it('shows memory bar when memory available', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ memory_percent: 72.3 } as any} />)
      expect(screen.getByText('72.3%')).toBeDefined()
      expect(screen.getByText('Memory')).toBeDefined()
    })
  })

  describe('model info', () => {
    it('shows Loaded when model loaded', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ model_loaded: true } as any} />)
      expect(screen.getByText('Loaded')).toBeDefined()
    })
    it('shows Not loaded when model not loaded', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ model_loaded: false } as any} />)
      expect(screen.getByText('Not loaded')).toBeDefined()
    })
    it('shows Loading... when model loading', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ model_loading: true } as any} />)
      expect(screen.getByText('Loading...')).toBeDefined()
    })
    it('shows model type', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ model_type: 'gpt2' } as any} />)
      expect(screen.getByText('gpt2')).toBeDefined()
    })
    it('shows parameters in B format', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ num_parameters: 1_500_000_000 } as any} />)
      expect(screen.getByText('1.5B')).toBeDefined()
    })
    it('shows parameters in M format', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ num_parameters: 124_000_000 } as any} />)
      expect(screen.getByText('124M')).toBeDefined()
    })
  })

  describe('inference stats', () => {
    it('shows request count', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ request_count: 42 } as any} />)
      expect(screen.getByText('42')).toBeDefined()
    })
    it('shows tokens/sec', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ tokens_per_sec: 15.3 } as any} />)
      expect(screen.getByText('15.3')).toBeDefined()
    })
    it('shows error count in red', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ error_count: 5 } as any} />)
      expect(screen.getByText('5')).toBeDefined()
    })
  })

  describe('uptime', () => {
    it('shows uptime from liveHealth', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ uptime_seconds: 7200 } as any} />)
      expect(screen.getByText(/2h/)).toBeDefined()
    })
    it('shows uptime from detailed', () => {
      render(<SystemHealthPanel {...emptyProps} detailed={{ uptime_seconds: 3600 } as any} />)
      expect(screen.getByText(/1h/)).toBeDefined()
    })
  })

  describe('expandable sections', () => {
    it('shows expandable for path latencies', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ path_latencies: [{ path: '/api/chat', count: 10, avg_ms: 5.2, p95_ms: 12.1 }] } as any} />)
      expect(screen.getByText('Endpoint Latency')).toBeDefined()
    })
    it('expands on click', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ path_latencies: [{ path: '/api/chat', count: 10, avg_ms: 5.2, p95_ms: 12.1 }] } as any} />)
      fireEvent.click(screen.getByText('Endpoint Latency'))
      expect(screen.getByText('/api/chat')).toBeDefined()
    })
    it('shows expandable for recent errors', () => {
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ recent_errors: [{ ts: '2026-01-01', status: 500, method: 'GET', path: '/api', error_type: 'Error', message: 'fail' }] } as any} />)
      expect(screen.getByText('Recent Errors')).toBeDefined()
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
      render(<SystemHealthPanel {...emptyProps} liveHealth={{ diagnoses: [{ check: 'memory', severity: 'warn', message: 'high usage' }] } as any} />)
      expect(screen.getByText('memory')).toBeDefined()
      expect(screen.getByText('high usage')).toBeDefined()
    })
  })

  describe('model health', () => {
    it('shows perplexity', () => {
      render(<SystemHealthPanel {...emptyProps} modelHealth={{ perplexity: 12.345 }} />)
      expect(screen.getByText('12.345')).toBeDefined()
    })
    it('shows loss', () => {
      render(<SystemHealthPanel {...emptyProps} modelHealth={{ loss: 0.5678 }} />)
      expect(screen.getByText('0.5678')).toBeDefined()
    })
    it('shows quality score as percentage', () => {
      render(<SystemHealthPanel {...emptyProps} modelHealth={{ quality_score: 0.85 }} />)
      expect(screen.getByText('85.0%')).toBeDefined()
    })
  })

  describe('sparklines', () => {
    it('renders sparklines when history has enough data', () => {
      const { container } = render(<SystemHealthPanel {...emptyProps} chartHistory={[{ time: '1', cpu: 10, mem: 20 }, { time: '2', cpu: 15, mem: 25 }, { time: '3', cpu: 12, mem: 22 }]} />)
      expect(container.querySelectorAll('svg').length).toBeGreaterThanOrEqual(1)
    })
  })
})
