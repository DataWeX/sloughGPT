import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

const { mockGet, mockPost } = vi.hoisted(() => ({
  mockGet: vi.fn(), mockPost: vi.fn(),
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
    Textarea: ({ value, onChange, placeholder }: any) => (
      <textarea value={value} onChange={onChange} placeholder={placeholder} />
    ),
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
  }
})

vi.mock('@/lib/http-client', () => ({
  apiGet: (...a: unknown[]) => mockGet(...a),
  apiPost: (...a: unknown[]) => mockPost(...a),
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.mock('@/components/composed/StatusBanner', () => ({
  StatusBanner: ({ message }: any) => <div data-testid="status-banner">{message}</div>,
}))

vi.mock('@/components/images/ImageGalleryInsightsCard', () => ({
  ImageGalleryInsightsCard: ({ gallery, styles }: any) => (
    <div data-testid="image-insights">
      {gallery?.length || 0} images, {styles?.length || 0} styles
    </div>
  ),
}))

import { ImagesPanel } from './ImagesPanel'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGet.mockImplementation((url: string) => {
    if (url.includes('gallery')) return Promise.resolve({ images: [] })
    if (url.includes('styles')) return Promise.resolve({ styles: [['realistic', 'Realistic'], ['cartoon', 'Cartoon']] })
    return Promise.resolve(null)
  })
  mockPost.mockResolvedValue({ image: 'base64data' })
})

describe('ImagesPanel — load flow (Create Images journey on /files)', () => {
  it('fetches gallery and styles on mount', async () => {
    render(<ImagesPanel />)
    await waitFor(() => {
      expect(mockGet).toHaveBeenCalledTimes(2)
    })
  })

  it('renders the generate card once loaded', async () => {
    render(<ImagesPanel />)
    await waitFor(() => {
      // "Generate" is both the card title and the button label — getAllByText.
      expect(screen.getAllByText('Generate').length).toBeGreaterThanOrEqual(2)
    })
  })

  it('renders nothing while the first load is pending', () => {
    mockGet.mockReturnValue(new Promise(() => {}))
    const { container } = render(<ImagesPanel />)
    expect(container.textContent).toBe('')
  })
})

describe('ImagesPanel — prompt flow', () => {
  it('renders prompt textarea after loading', async () => {
    render(<ImagesPanel />)
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('allows typing a prompt', async () => {
    render(<ImagesPanel />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1) })

    const textarea = screen.getAllByPlaceholderText(/describe/i)[0]
    fireEvent.change(textarea, { target: { value: 'A sunset over mountains' } })
    expect((textarea as HTMLTextAreaElement).value).toBe('A sunset over mountains')
  })
})

describe('ImagesPanel — style selection flow', () => {
  it('displays available styles', async () => {
    render(<ImagesPanel />)
    await waitFor(() => {
      expect(screen.getByText('Realistic')).toBeTruthy()
      expect(screen.getByText('Cartoon')).toBeTruthy()
    })
  })

  it('allows selecting a style', async () => {
    render(<ImagesPanel />)
    await waitFor(() => { expect(screen.getByText('Realistic')).toBeTruthy() })

    fireEvent.click(screen.getByText('Cartoon'))
    expect(screen.getByText('Cartoon')).toBeTruthy()
  })
})

describe('ImagesPanel — generate flow', () => {
  it('generate button triggers image generation', async () => {
    render(<ImagesPanel />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1) })

    const textarea = screen.getAllByPlaceholderText(/describe/i)[0]
    fireEvent.change(textarea, { target: { value: 'A cat' } })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    expect(genBtn).toBeTruthy()

    await act(async () => { fireEvent.click(genBtn!) })
    await waitFor(() => {
      expect(mockPost).toHaveBeenCalledWith(
        expect.stringContaining('generate'),
        expect.objectContaining({ prompt: 'A cat' })
      )
    })
  })

  it('empty prompt does not trigger generation', async () => {
    render(<ImagesPanel />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1) })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    if (genBtn) {
      fireEvent.click(genBtn)
      expect(mockPost).not.toHaveBeenCalled()
    }
  })
})

describe('ImagesPanel — gallery display', () => {
  it('shows empty gallery state', async () => {
    render(<ImagesPanel />)
    await waitFor(() => {
      expect(screen.getByText(/no images generated/i)).toBeTruthy()
    })
  })

  it('displays images when gallery has items', async () => {
    mockGet.mockImplementation((url: string) => {
      if (url.includes('gallery')) return Promise.resolve({
        images: [{ id: '1', path: '/img1.png', created: Date.now() }]
      })
      if (url.includes('styles')) return Promise.resolve({ styles: [] })
      return Promise.resolve(null)
    })
    render(<ImagesPanel />)
    await waitFor(() => {
      expect(screen.getByTestId('image-insights')).toBeTruthy()
    })
  })

  it('insights card reports the gallery contents', async () => {
    mockGet.mockImplementation((url: string) => {
      if (url.includes('gallery')) return Promise.resolve({
        images: [{ id: '1', path: '/a.png', created: Date.now() }, { id: '2', path: '/b.png', created: Date.now() }]
      })
      if (url.includes('styles')) return Promise.resolve({ styles: [['realistic', 'Realistic']] })
      return Promise.resolve(null)
    })
    render(<ImagesPanel />)
    await waitFor(() => {
      expect(screen.getByTestId('image-insights').textContent).toContain('2 images, 1 styles')
    })
  })
})

describe('ImagesPanel — error handling', () => {
  it('handles gallery load failure gracefully', async () => {
    mockGet.mockRejectedValue(new Error('network'))
    render(<ImagesPanel />)
    await waitFor(() => {
      expect(screen.getByTestId('status-banner').textContent).toContain('Could not load image data')
    })
  })

  it('handles generation failure gracefully', async () => {
    mockPost.mockRejectedValue(new Error('generation failed'))
    render(<ImagesPanel />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/describe/i).length).toBeGreaterThanOrEqual(1) })

    const textarea = screen.getAllByPlaceholderText(/describe/i)[0]
    fireEvent.change(textarea, { target: { value: 'A cat' } })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    await act(async () => { fireEvent.click(genBtn!) })
    await waitFor(() => {
      expect(screen.getByTestId('status-banner').textContent).toContain('generation failed')
    })
  })
})

describe('ImagesPanel — insights gating', () => {
  it('does not render insights card when gallery is empty', async () => {
    render(<ImagesPanel />)
    await waitFor(() => {
      expect(screen.queryByTestId('image-insights')).toBeNull()
    })
  })
})
