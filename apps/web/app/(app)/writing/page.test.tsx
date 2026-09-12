import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/writing',
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

import WritingAssistantPage from './page'
import { generateTool } from '@/lib/tools-controller'
import type { ToolProfile } from '@/lib/tools-controller'

const writingProfile: ToolProfile = {
  id: 'writing',
  name: 'Writing Assistant',
  description: 'desc',
  icon: 'sparkle',
  params: [],
  options: {
    tone: [
      { id: 'witty', label: 'Witty', description: '' },
      { id: 'bold', label: 'Bold', description: '' },
    ],
    type: [
      { id: 'essay', label: 'Essay', description: '' },
      { id: 'haiku', label: 'Haiku', description: '' },
    ],
  },
  default_options: { tone: 'witty', type: 'essay' },
  system_prompt: 'sp',
  max_tokens: 700,
}

describe('WritingAssistantPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseToolProfile.mockReturnValue(null)
  })

  it('renders the page title', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByText('Writing Assistant')).toBeDefined()
  })

  it('has tone selection buttons', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByText('Friendly')).toBeDefined()
    expect(screen.getByText('Professional')).toBeDefined()
    expect(screen.getByText('Funny')).toBeDefined()
    expect(screen.getByText('Short')).toBeDefined()
    expect(screen.getByText('Detailed')).toBeDefined()
  })

  it('has type selection buttons', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByText('Email')).toBeDefined()
    expect(screen.getByText('Social Post')).toBeDefined()
    expect(screen.getByText('Story')).toBeDefined()
    expect(screen.getByText('Poem')).toBeDefined()
    expect(screen.getByText('Letter')).toBeDefined()
    expect(screen.getByText('Note')).toBeDefined()
  })

  it('has a textarea for input', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByPlaceholderText(/tell me what you want to write about/i)).toBeDefined()
  })

  it('disables write button when input is empty', () => {
    render(<WritingAssistantPage />)
    
    const writeButton = screen.getByRole('button', { name: /^Write$/i })
    expect(writeButton).toBeDisabled()
  })

  it('calls generateTool when writing with input', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<WritingAssistantPage />)
    
    const inputTextarea = screen.getByPlaceholderText(/tell me what you want to write about/i)
    fireEvent.change(inputTextarea, { target: { value: 'Write a professional email about...' } })
    
    const writeButton = screen.getByRole('button', { name: /^Write$/i })
    fireEvent.click(writeButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'writing',
        expect.objectContaining({
          text: 'Write a professional email about...',
          action: 'write',
          tone: 'professional',
          type: 'email',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })

  it('allows changing tone and type', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<WritingAssistantPage />)
    
    const friendlyButton = screen.getByText('Friendly')
    fireEvent.click(friendlyButton)
    
    const storyButton = screen.getByText('Story')
    fireEvent.click(storyButton)
    
    const inputTextarea = screen.getByPlaceholderText(/tell me what you want to write about/i)
    fireEvent.change(inputTextarea, { target: { value: 'Write a story about...' } })
    
    const writeButton = screen.getByRole('button', { name: /^Write$/i })
    fireEvent.click(writeButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'writing',
        expect.objectContaining({
          tone: 'friendly',
          type: 'story',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })

  it('renders option buttons from the backend profile when available', () => {
    mockUseToolProfile.mockReturnValue(writingProfile)

    render(<WritingAssistantPage />)

    expect(screen.getByText('Witty')).toBeDefined()
    expect(screen.getByText('Bold')).toBeDefined()
    expect(screen.getByText('Essay')).toBeDefined()
    expect(screen.getByText('Haiku')).toBeDefined()
    expect(screen.queryByText('Friendly')).toBeNull()
  })
})
