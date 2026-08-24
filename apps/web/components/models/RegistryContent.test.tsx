// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/registry-controller', () => ({
  registryController: { list: vi.fn(), stats: vi.fn(), best: vi.fn() },
}))
vi.mock('@/components/registry/RegistryHealthCard', () => ({
  RegistryHealthCard: () => <div data-testid="registry-health" />,
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: vi.fn() }),
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  StatCard: ({ label, value, ...p }: any) => <div data-testid="stat-card" data-label={label}>{String(value)}</div>,
  KpiGrid: ({ children, ...p }: any) => <div data-testid="kpi-grid" {...p}>{children}</div>,
  SearchInput: ({ value, onChange, ...p }: any) => <input value={value} onChange={e => onChange(e.target.value)} {...p} />,
  Skeleton: (p: any) => <div data-testid="skeleton" {...p} />,
  IconRefresh: () => <span>↻</span>,
}))

import RegistryContent from './RegistryContent'
import { registryController } from '@/lib/registry-controller'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(registryController.list).mockResolvedValue([
    { name: 'gpt2', status: 'loaded', type: 'local' },
    { name: 'qwen', status: 'failed', type: 'remote', error: 'Timeout' },
  ] as any)
  vi.mocked(registryController.stats).mockResolvedValue({ total: 2, loaded: 1, failed: 1, circuit_breaker: 'closed' } as any)
  vi.mocked(registryController.best).mockResolvedValue({ name: 'gpt2' } as any)
})

afterEach(() => cleanup())

describe('RegistryContent', () => {
  it('calls registry APIs on mount', async () => {
    render(<RegistryContent />)
    await waitFor(() => {
      expect(registryController.list).toHaveBeenCalled()
      expect(registryController.stats).toHaveBeenCalled()
      expect(registryController.best).toHaveBeenCalled()
    })
  })

  it('displays KPI stats', async () => {
    render(<RegistryContent />)
    await waitFor(() => {
      expect(screen.getAllByTestId('stat-card').length).toBeGreaterThanOrEqual(4)
    })
  })
})
