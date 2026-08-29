// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import EngineStatusCard from './EngineStatusCard'

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    getEngineStatus: vi.fn(),
    reloadEngine: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

import { modelController } from '@/lib/model-controller'

describe('EngineStatusCard', () => {
  afterEach(cleanup)
  beforeEach(() => { vi.clearAllMocks() })

  it('renders engine status', async () => {
    vi.mocked(modelController.getEngineStatus).mockResolvedValue({
      engine: 'slo', version: '1.0.0', models_loaded: 1, uptime_s: 3600, memory_usage_mb: 512,
    })
    render(<EngineStatusCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('slo')).toBeDefined()
      expect(screen.getByText('1.0.0')).toBeDefined()
      expect(screen.getByText('1')).toBeDefined()
      expect(screen.getByText('1h 0m')).toBeDefined()
      expect(screen.getByText('512 MB')).toBeDefined()
    })
  })

  it('shows reload button', async () => {
    vi.mocked(modelController.getEngineStatus).mockResolvedValue({
      engine: 'slo', version: '1.0.0', models_loaded: 0, uptime_s: 0, memory_usage_mb: 0,
    })
    render(<EngineStatusCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('Reload')).toBeDefined()
    })
  })

  it('formats uptime in minutes', async () => {
    vi.mocked(modelController.getEngineStatus).mockResolvedValue({
      engine: 'slo', version: '1.0.0', models_loaded: 0, uptime_s: 120, memory_usage_mb: 0,
    })
    render(<EngineStatusCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('2m')).toBeDefined()
    })
  })

  it('formats uptime in seconds', async () => {
    vi.mocked(modelController.getEngineStatus).mockResolvedValue({
      engine: 'slo', version: '1.0.0', models_loaded: 0, uptime_s: 30, memory_usage_mb: 0,
    })
    render(<EngineStatusCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('30s')).toBeDefined()
    })
  })

  it('shows skeleton loading state', () => {
    vi.mocked(modelController.getEngineStatus).mockReturnValue(new Promise(() => {}))
    const { container } = render(<EngineStatusCard />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(1)
  })

  it('hides when status is null', async () => {
    vi.mocked(modelController.getEngineStatus).mockResolvedValue(null as never)
    const { container } = render(<EngineStatusCard />)
    await vi.waitFor(() => {
      expect(container.querySelector('[class*="rounded-lg"]')).toBeNull()
    })
  })
})
