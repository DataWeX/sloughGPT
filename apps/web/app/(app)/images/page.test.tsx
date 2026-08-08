import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockGet = vi.fn()
const mockPost = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockGet(...args),
  apiPost: (...args: unknown[]) => mockPost(...args),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import ImagesPage from './page'

describe('ImagesPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGet.mockImplementation((url: string) => {
      if (url.includes('gallery')) return Promise.resolve({ data: { images: [] } })
      if (url.includes('styles')) return Promise.resolve({ data: { styles: [['realistic', 'Realistic'], ['cartoon', 'Cartoon']] } })
      return Promise.resolve(null)
    })
  })

  it('renders page header', async () => {
    render(<ImagesPage />)
    expect(screen.getAllByText('Images').length).toBeGreaterThanOrEqual(1)
  })

  it('renders prompt textarea after loading', async () => {
    render(<ImagesPage />)
    await screen.findByPlaceholderText(/describe/i)
    expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders generate button after loading', async () => {
    render(<ImagesPage />)
    await screen.findAllByText('No images generated yet.')
    expect(screen.getAllByText('Generate').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty gallery state', async () => {
    render(<ImagesPage />)
    await screen.findAllByText(/no images|empty/i)
  })
})
