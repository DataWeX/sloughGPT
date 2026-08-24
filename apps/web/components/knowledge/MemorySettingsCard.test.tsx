// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/memory-controller', () => ({
  memoryController: {
    getConfig: vi.fn(),
    setEnabled: vi.fn(),
    consolidate: vi.fn(),
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: vi.fn() }),
}))
vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: any) => e?.message || 'Unknown error',
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Switch: ({ checked, onCheckedChange, ...p }: any) => <input type="checkbox" checked={checked} onChange={e => onCheckedChange?.(e.target.checked)} {...p} />,
  Skeleton: (p: any) => <div data-testid="skeleton" {...p} />,
}))

import { MemorySettingsCard } from './MemorySettingsCard'
import { memoryController } from '@/lib/memory-controller'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(memoryController.getConfig).mockResolvedValue({
    enabled: true, max_facts: 100, min_chars: 50,
    archive_retention_days: 30, consolidation_threshold: 10,
  } as any)
  vi.mocked(memoryController.setEnabled).mockResolvedValue(undefined as any)
  vi.mocked(memoryController.consolidate).mockResolvedValue(undefined as any)
})

afterEach(() => cleanup())

describe('MemorySettingsCard', () => {
  it('calls getConfig on mount', async () => {
    render(<MemorySettingsCard />)
    await waitFor(() => {
      expect(memoryController.getConfig).toHaveBeenCalled()
    })
  })

  it('toggles enabled state', async () => {
    render(<MemorySettingsCard />)
    await waitFor(() => {
      expect(screen.getByRole('checkbox')).toBeDefined()
    })
    fireEvent.click(screen.getByRole('checkbox'))
    await waitFor(() => {
      expect(memoryController.setEnabled).toHaveBeenCalledWith(false)
    })
  })

  it('runs consolidation', async () => {
    render(<MemorySettingsCard />)
    await waitFor(() => {
      expect(screen.getByText('Run Consolidation Now')).toBeDefined()
    })
    fireEvent.click(screen.getByText('Run Consolidation Now'))
    await waitFor(() => {
      expect(memoryController.consolidate).toHaveBeenCalled()
    })
  })
})
