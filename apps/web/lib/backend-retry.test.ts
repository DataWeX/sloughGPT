import { describe, it, expect, vi, beforeEach } from 'vitest'

const mocks = vi.hoisted(() => ({
  apiGet: vi.fn(),
  revalidateAll: vi.fn(),
}))

vi.mock('@/lib/http-client', () => ({ apiGet: mocks.apiGet }))
vi.mock('@/lib/cache', () => ({ revalidateAll: mocks.revalidateAll }))

import { probeBackendAndRevalidate } from './backend-retry'

/**
 * One retry path for every surface (card 87345605): the status-bar Retry
 * button, the startup overlay's stuck-screen Retry and the backend-down
 * banner action all probe silently and revalidate in place. Before the fix
 * the overlay called location.reload() while the status bar probed — two
 * surfaces, two behaviours, and the reload threw away client state.
 */
describe('probeBackendAndRevalidate (card 87345605)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('probes /health silently and revalidates cached views on success', async () => {
    mocks.apiGet.mockResolvedValue({ status: 'ok' })
    await expect(probeBackendAndRevalidate()).resolves.toBe(true)
    expect(mocks.apiGet).toHaveBeenCalledWith(
      '/health',
      undefined,
      expect.objectContaining({ silent: true }),
    )
    expect(mocks.revalidateAll).toHaveBeenCalledTimes(1)
  })

  it('does not revalidate and does not throw when the probe fails', async () => {
    mocks.apiGet.mockRejectedValue(new Error('connection refused'))
    await expect(probeBackendAndRevalidate()).resolves.toBe(false)
    expect(mocks.revalidateAll).not.toHaveBeenCalled()
  })
})
