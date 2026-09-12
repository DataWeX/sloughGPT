import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  StatCard: ({ label, value, loading }: any) => <div>{loading ? 'Loading...' : `${label}: ${value}`}</div>,
  KpiGrid: ({ children }: any) => <div>{children}</div>,
  IconRefresh: ({ className }: any) => <span className={className} />,
  Spinner: ({ className }: any) => <span className={className} />,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: any) => selector({ addToast: mockAddToast }),
}))

vi.mock('@/components/composed/StatusBanner', () => ({
  StatusBanner: ({ variant, message }: { variant: string; message: string }) => (
    <div data-variant={variant}>{message}</div>
  ),
}))

import { InferencePoolCard } from './InferencePoolCard'

vi.mock('@/lib/system-controller', () => ({
  systemController: {
    getInferencePoolStatus: vi.fn(),
  },
}))

import { systemController } from '@/lib/system-controller'

afterEach(() => { cleanup(); vi.clearAllMocks() })

it('renders loading state while loading', () => {
  vi.mocked(systemController.getInferencePoolStatus).mockReturnValue(new Promise(() => {}))
  const { container } = render(<InferencePoolCard />)
  expect(container.textContent).toContain('Loading...')
})

it('shows error banner when fetch fails', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockRejectedValue(new Error('fail'))
  render(<InferencePoolCard />)
  expect(await screen.findByText('Failed to load')).toBeTruthy()
})

it('shows Yes when initialized', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true, max_workers: 4 })
  render(<InferencePoolCard />)
  expect(await screen.findByText('Initialized: Yes')).toBeTruthy()
  expect(screen.getByText('Inference Pool')).toBeTruthy()
})

it('shows No when not initialized', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: false })
  render(<InferencePoolCard />)
  expect(await screen.findByText('Initialized: No')).toBeTruthy()
})

it('shows max_workers when present', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true, max_workers: 8 })
  render(<InferencePoolCard />)
  expect(await screen.findByText('Max Workers: 8')).toBeTruthy()
})

it('shows queue_timeout when present', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true, queue_timeout: 5.2 })
  render(<InferencePoolCard />)
  expect(await screen.findByText('Queue Timeout: 5.2ms')).toBeTruthy()
})

it('shows error message when present', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: false, error: 'OOM killed' })
  render(<InferencePoolCard />)
  expect(await screen.findByText('OOM killed')).toBeTruthy()
})

it('hides error when absent', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true })
  render(<InferencePoolCard />)
  await vi.waitFor(() => expect(screen.queryByText('Initialized: Yes')).toBeTruthy())
  expect(screen.queryByText('OOM killed')).toBeNull()
})

it('calls onRefresh when Refresh clicked', async () => {
  const onRefresh = vi.fn()
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true })
  render(<InferencePoolCard onRefresh={onRefresh} />)
  await vi.waitFor(() => expect(screen.getByText('Initialized: Yes')).toBeTruthy())
  fireEvent.click(screen.getByRole('button', { name: /refresh inference pool/i }))
  expect(onRefresh).toHaveBeenCalledOnce()
})

it('does not crash when Refresh clicked and onRefresh is absent', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true })
  render(<InferencePoolCard />)
  await vi.waitFor(() => expect(screen.getByText('Initialized: Yes')).toBeTruthy())
  const btn = screen.getByRole('button', { name: /refresh inference pool/i })
  expect(btn).toBeTruthy()
  fireEvent.click(btn)
  await vi.waitFor(() => expect(screen.getByText('Initialized: Yes')).toBeTruthy())
})

it('shows — when max_workers is null', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true, max_workers: undefined })
  render(<InferencePoolCard />)
  expect(await screen.findByText('Max Workers: —')).toBeTruthy()
})
