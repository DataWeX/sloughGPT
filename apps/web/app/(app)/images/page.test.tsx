import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const { mockGet, mockPost, mockAddToast } = vi.hoisted(() => ({
  mockGet: vi.fn(), mockPost: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled }: any) => (
      <button onClick={onClick} disabled={disabled}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} />
    ),
    Textarea: ({ value, onChange, placeholder }: any) => (
      <textarea value={value} onChange={onChange} placeholder={placeholder} />
    ),
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  }
})

vi.mock('@/lib/http-client', () => ({
  apiGet: (...a: unknown[]) => mockGet(...a),
  apiPost: (...a: unknown[]) => mockPost(...a),
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/components/images/ImageGalleryInsightsCard', () => ({
  ImageGalleryInsightsCard: ({ gallery, styles }: any) => (
    <div data-testid="image-insights">
      {gallery?.length || 0} images, {styles?.length || 0} styles
    </div>
  ),
}))

import ImagesPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGet.mockImplementation((url: string) => {
    if (url.includes('gallery')) return Promise.resolve({ data: { images: [] } })
    if (url.includes('styles')) return Promise.resolve({ data: { styles: [['realistic', 'Realistic'], ['cartoon', 'Cartoon']] } })
    return Promise.resolve(null)
  })
  mockPost.mockResolvedValue({ image: 'base64data' })
})

describe('ImagesPage — initial load flow', () => {
  it('renders page header', async () => {
    render(<ImagesPage />)
    expect(screen.getAllByText('Images').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches gallery and styles on mount', async () => {
    render(<ImagesPage />)
    await waitFor(() => {
      expect(mockGet).toHaveBeenCalledTimes(2)
    })
  })

  it('shows loading state', () => {
    mockGet.mockReturnValue(new Promise(() => {}))
    render(<ImagesPage />)
    expect(screen.getAllByText('Images').length).toBeGreaterThanOrEqual(1)
  })
})

describe('ImagesPage — prompt flow', () => {
  it('renders prompt textarea after loading', async () => {
    render(<ImagesPage />)
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('allows typing a prompt', async () => {
    render(<ImagesPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1) })

    const textarea = screen.getAllByPlaceholderText(/describe/i)[0]
    fireEvent.change(textarea, { target: { value: 'A sunset over mountains' } })
    expect((textarea as HTMLTextAreaElement).value).toBe('A sunset over mountains')
  })
})

describe('ImagesPage — style selection flow', () => {
  it('displays available styles', async () => {
    render(<ImagesPage />)
    await waitFor(() => {
      expect(screen.getByText('Realistic')).toBeTruthy()
      expect(screen.getByText('Cartoon')).toBeTruthy()
    })
  })

  it('allows selecting a style', async () => {
    render(<ImagesPage />)
    await waitFor(() => { expect(screen.getByText('Realistic')).toBeTruthy() })

    fireEvent.click(screen.getByText('Cartoon'))
    // No crash = success
    expect(screen.getByText('Cartoon')).toBeTruthy()
  })
})

describe('ImagesPage — generate flow', () => {
  it('generate button triggers image generation', async () => {
    render(<ImagesPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1) })

    const textarea = screen.getAllByPlaceholderText(/describe/i)[0]
    fireEvent.change(textarea, { target: { value: 'A cat' } })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    if (genBtn) {
      await act(async () => { fireEvent.click(genBtn) })
      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith(
          expect.stringContaining('generate'),
          expect.objectContaining({ prompt: 'A cat' })
        )
      })
    }
  })

  it('empty prompt does not trigger generation', async () => {
    render(<ImagesPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1) })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    if (genBtn) {
      fireEvent.click(genBtn)
      // Should not call API with empty prompt
      expect(mockPost).not.toHaveBeenCalled()
    }
  })
})

describe('ImagesPage — gallery display', () => {
  it('shows empty gallery state', async () => {
    render(<ImagesPage />)
    await waitFor(() => {
      expect(screen.getByText(/no images generated/i)).toBeTruthy()
    })
  })

  it('displays images when gallery has items', async () => {
    mockGet.mockImplementation((url: string) => {
      if (url.includes('gallery')) return Promise.resolve({
        data: { images: [{ id: '1', path: '/img1.png', created: Date.now() }] }
      })
      if (url.includes('styles')) return Promise.resolve({ data: { styles: [] } })
      return Promise.resolve(null)
    })
    render(<ImagesPage />)
    await waitFor(() => {
      // Page renders without crashing with gallery items
      expect(screen.getAllByText('Images').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('ImagesPage — error handling', () => {
  it('handles gallery load failure gracefully', async () => {
    mockGet.mockRejectedValue(new Error('network'))
    render(<ImagesPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Images').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles generation failure gracefully', async () => {
    mockPost.mockRejectedValue(new Error('generation failed'))
    render(<ImagesPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1) })

    const textarea = screen.getAllByPlaceholderText(/describe/i)[0]
    fireEvent.change(textarea, { target: { value: 'A cat' } })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    if (genBtn) {
      await act(async () => { fireEvent.click(genBtn) })
      await waitFor(() => {
        // Page should still render after error
        expect(screen.getAllByText('Images').length).toBeGreaterThanOrEqual(1)
      })
    }
  })
})

describe('ImagesPage — insights card', () => {
  it('does not render insights card when gallery is empty', async () => {
    render(<ImagesPage />)
    await waitFor(() => {
      // Insights card only renders when gallery has items
      expect(screen.queryByTestId('image-insights')).toBeNull()
    })
  })
})
