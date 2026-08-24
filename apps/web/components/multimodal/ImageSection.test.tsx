// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/images-controller', () => ({
  imagesController: { gallery: vi.fn(), styles: vi.fn() },
}))
vi.mock('@/lib/http-client', () => ({
  apiPost: vi.fn(),
}))
vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))
vi.mock('@/components/images/ImageGalleryInsightsCard', () => ({
  ImageGalleryInsightsCard: () => <div data-testid="insights" />,
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
  Textarea: ({ value, onChange, ...p }: any) => <textarea value={value} onChange={e => onChange(e.target.value)} {...p} />,
  IconRefresh: () => <span>↻</span>,
}))

import { ImageSection } from './ImageSection'
import { imagesController } from '@/lib/images-controller'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(imagesController.gallery).mockResolvedValue([] as any)
  vi.mocked(imagesController.styles).mockResolvedValue([] as any)
})

afterEach(() => cleanup())

describe('ImageSection', () => {
  it('calls gallery and styles on mount', async () => {
    render(<ImageSection />)
    await waitFor(() => {
      expect(imagesController.gallery).toHaveBeenCalled()
      expect(imagesController.styles).toHaveBeenCalled()
    })
  })

  it('renders without crashing', async () => {
    render(<ImageSection />)
    await waitFor(() => {
      expect(imagesController.gallery).toHaveBeenCalled()
    })
  })
})
