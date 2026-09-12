import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/explain',
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/tools-controller', () => ({
  generateTool: vi.fn(),
}))

import ExplainPage from './page'
import { generateTool } from '@/lib/tools-controller'

describe('ExplainPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<ExplainPage />)
    expect(screen.getByText('Explain Things Simply')).toBeDefined()
  })

  it('has a textarea for topic input', () => {
    render(<ExplainPage />)
    expect(screen.getByPlaceholderText(/e\.g\. How does the internet work/i)).toBeDefined()
  })

  it('has difficulty selection buttons', () => {
    render(<ExplainPage />)
    expect(screen.getByText('Simple')).toBeDefined()
    expect(screen.getByText('Normal')).toBeDefined()
    expect(screen.getByText('Detailed')).toBeDefined()
  })

  it('has an explain button', () => {
    render(<ExplainPage />)
    expect(screen.getByRole('button', { name: /^Explain$/ })).toBeDefined()
  })

  it('shows toast when submitting empty form', async () => {
    render(<ExplainPage />)
    
    const explainButton = screen.getByRole('button', { name: /^Explain$/ })
    fireEvent.click(explainButton)

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Type something to explain', 'info')
    })
  })

  it('calls generateTool when topic is provided', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<ExplainPage />)
    
    const topicInput = screen.getByPlaceholderText(/e\.g\. How does the internet work/i)
    fireEvent.change(topicInput, { target: { value: 'How does the internet work?' } })
    
    const explainButton = screen.getByRole('button', { name: /^Explain$/ })
    fireEvent.click(explainButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'explain',
        expect.objectContaining({
          topic: 'How does the internet work?',
          difficulty: 'normal',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })

  it('allows changing difficulty', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<ExplainPage />)
    
    const simpleButton = screen.getByText('Simple')
    fireEvent.click(simpleButton)
    
    const topicInput = screen.getByPlaceholderText(/e\.g\. How does the internet work/i)
    fireEvent.change(topicInput, { target: { value: 'Test topic' } })
    
    const explainButton = screen.getByRole('button', { name: /^Explain$/ })
    fireEvent.click(explainButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'explain',
        expect.objectContaining({
          difficulty: 'simple',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })
})
