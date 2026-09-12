import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/rewrite',
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

import RewritePage from './page'
import { generateTool } from '@/lib/tools-controller'
import type { ToolProfile } from '@/lib/tools-controller'

const rewriteProfile: ToolProfile = {
  id: 'rewrite',
  name: 'Rewrite & Polish',
  description: 'desc',
  icon: 'sparkle',
  params: [],
  options: {
    action: [
      { id: 'grammar', label: 'Fix Grammar', description: '' },
      { id: 'shorter', label: 'Make Shorter', description: '' },
      { id: 'friendlier', label: 'Make Friendlier', description: '' },
      { id: 'professional', label: 'Make Professional', description: '' },
      { id: 'sound-like-me', label: 'Sound Like Me', description: '' },
    ],
  },
  default_options: { action: 'grammar' },
  system_prompt: 'sp',
  max_tokens: 700,
}

describe('RewritePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseToolProfile.mockReturnValue(null)
  })

  it('renders the page title', () => {
    render(<RewritePage />)
    expect(screen.getByText('Rewrite & Polish')).toBeDefined()
  })

  it('has a textarea for original text', () => {
    render(<RewritePage />)
    expect(screen.getByPlaceholderText(/paste what you wrote/i)).toBeDefined()
  })

  it('has action buttons', () => {
    render(<RewritePage />)
    expect(screen.getByText('Fix Grammar')).toBeDefined()
    expect(screen.getByText('Make Shorter')).toBeDefined()
    expect(screen.getByText('Make Friendlier')).toBeDefined()
    expect(screen.getByText('Make Professional')).toBeDefined()
    expect(screen.getByText('Sound Like Me')).toBeDefined()
  })

  it('disables action buttons with empty text', async () => {
    render(<RewritePage />)
    
    const grammarButton = screen.getByText('Fix Grammar')
    expect(grammarButton.hasAttribute('disabled')).toBe(true)
  })

  it('calls generateTool when action is clicked with text', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<RewritePage />)
    
    const originalTextarea = screen.getByPlaceholderText(/paste what you wrote/i)
    fireEvent.change(originalTextarea, { target: { value: 'Test text to rewrite' } })
    
    const grammarButton = screen.getByText('Fix Grammar')
    fireEvent.click(grammarButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'rewrite',
        expect.objectContaining({
          text: 'Test text to rewrite',
          action: 'grammar',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })

  it('renders option buttons from backend profile when available', () => {
    mockUseToolProfile.mockReturnValue(rewriteProfile)

    render(<RewritePage />)

    expect(screen.getByText('Fix Grammar')).toBeDefined()
    expect(screen.getByText('Make Shorter')).toBeDefined()
    expect(screen.getByText('Make Friendlier')).toBeDefined()
    expect(screen.getByText('Make Professional')).toBeDefined()
    expect(screen.getByText('Sound Like Me')).toBeDefined()
  })
})
