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

const { mockUseToolProfile } = vi.hoisted(() => ({
  mockUseToolProfile: vi.fn(() => null as ToolProfile | null),
}))

vi.mock('@/lib/use-tool-profile', () => ({
  useToolProfile: mockUseToolProfile,
}))

import ExplainPage from './page'
import { generateTool } from '@/lib/tools-controller'
import type { ToolProfile } from '@/lib/tools-controller'

const explainProfile: ToolProfile = {
  id: 'explain',
  name: 'Explain Things Simply',
  description: 'desc',
  icon: 'search',
  params: [],
  options: {
    difficulty: [
      { id: 'simple', label: 'Simple', description: '' },
      { id: 'normal', label: 'Normal', description: '' },
      { id: 'detailed', label: 'Detailed', description: '' },
    ],
  },
  default_options: { difficulty: 'normal' },
  system_prompt: 'sp',
  max_tokens: 900,
}

describe('ExplainPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseToolProfile.mockReturnValue(null)
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

it('disables the explain button with empty form', async () => {
    render(<ExplainPage />)
    
    const explainButton = screen.getByRole('button', { name: /^Explain$/ })
    expect(explainButton.hasAttribute('disabled')).toBe(true)
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

  it('renders option buttons from backend profile when available', () => {
    mockUseToolProfile.mockReturnValue(explainProfile)

    render(<ExplainPage />)

    expect(screen.getByText('Simple')).toBeDefined()
    expect(screen.getByText('Normal')).toBeDefined()
    expect(screen.getByText('Detailed')).toBeDefined()
  })
})
