import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { InferencePoolCard } from './InferencePoolCard'

vi.mock('@/lib/system-controller', () => ({
  systemController: {
    getInferencePoolStatus: vi.fn(),
  },
}))

import { systemController } from '@/lib/system-controller'

afterEach(() => { cleanup(); vi.resetAllMocks() })

it('renders nothing while loading', () => {
  vi.mocked(systemController.getInferencePoolStatus).mockReturnValue(new Promise(() => {}))
  const { container } = render(<InferencePoolCard />)
  expect(container.innerHTML).toBe('')
})

it('renders nothing when status is null', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockRejectedValue(new Error('fail'))
  const { container } = render(<InferencePoolCard />)
  await vi.waitFor(() => expect(container.innerHTML).toBe(''))
})

it('shows Active when initialized', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true, max_workers: 4 })
  render(<InferencePoolCard />)
  expect(await screen.findByText('Active')).toBeTruthy()
  expect(screen.getByText('Inference Pool')).toBeTruthy()
})

it('shows Inactive when not initialized', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: false })
  render(<InferencePoolCard />)
  expect(await screen.findByText('Inactive')).toBeTruthy()
})

it('shows max_workers when present', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true, max_workers: 8 })
  render(<InferencePoolCard />)
  expect(await screen.findByText('8')).toBeTruthy()
})

it('shows queue_timeout when present', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true, queue_timeout: 5.2 })
  render(<InferencePoolCard />)
  expect(await screen.findByText('5.2s')).toBeTruthy()
})

it('shows error message when present', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: false, error: 'OOM killed' })
  render(<InferencePoolCard />)
  expect(await screen.findByText('OOM killed')).toBeTruthy()
})

it('hides error when absent', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true })
  render(<InferencePoolCard />)
  await vi.waitFor(() => expect(screen.queryByText('Active')).toBeTruthy())
  expect(screen.queryByText('OOM killed')).toBeNull()
})

it('calls onRefresh and refetches when Refresh clicked', async () => {
  const onRefresh = vi.fn()
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true })
  render(<InferencePoolCard onRefresh={onRefresh} />)
  await vi.waitFor(() => expect(screen.getByText('Active')).toBeTruthy())
  screen.getByText('Refresh').click()
  expect(onRefresh).toHaveBeenCalledOnce()
  expect(systemController.getInferencePoolStatus).toHaveBeenCalledTimes(2)
})

it('does not render Refresh button when onRefresh is absent', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true })
  render(<InferencePoolCard />)
  await vi.waitFor(() => expect(screen.getByText('Active')).toBeTruthy())
  expect(screen.queryByText('Refresh')).toBeNull()
})

it('hides max_workers when null', async () => {
  vi.mocked(systemController.getInferencePoolStatus).mockResolvedValue({ initialized: true, max_workers: null })
  render(<InferencePoolCard />)
  await vi.waitFor(() => expect(screen.getByText('Active')).toBeTruthy())
  expect(screen.queryByText('Max Workers')).toBeNull()
})
