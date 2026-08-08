import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import SecurityPage from './page'

describe('SecurityPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue({ data: { logs: [] } })
  })

  it('renders page header', async () => {
    render(<SecurityPage />)
    expect(screen.getAllByText('Security').length).toBeGreaterThanOrEqual(1)
  })

  it('renders search input after loading', async () => {
    render(<SecurityPage />)
    await screen.findByPlaceholderText(/filter by event/i)
    expect(screen.getAllByPlaceholderText(/filter by event/i).length).toBeGreaterThanOrEqual(1)
  })
})
