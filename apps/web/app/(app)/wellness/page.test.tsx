import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/wellness',
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

import WellnessPage from './page'
import { generateTool } from '@/lib/tools-controller'

describe('WellnessPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<WellnessPage />)
    expect(screen.getByText('Make Me Well')).toBeDefined()
  })

  it('shows wellness options', () => {
    render(<WellnessPage />)
    expect(screen.getByText('Sleep Story')).toBeDefined()
    expect(screen.getByText('Meditation')).toBeDefined()
    expect(screen.getByText('Journal Prompt')).toBeDefined()
    expect(screen.getByText('Breathing Exercise')).toBeDefined()
    expect(screen.getByText('Positive Affirmation')).toBeDefined()
  })

  it('shows preferences input when option is selected', async () => {
    render(<WellnessPage />)
    
    const sleepOption = screen.getByText('Sleep Story')
    fireEvent.click(sleepOption)

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/e\.g\. A story about the ocean/i)).toBeDefined()
    })
  })

  it('calls generateTool when generating with option selected', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<WellnessPage />)
    
    const sleepOption = screen.getByText('Sleep Story')
    fireEvent.click(sleepOption)
    
    const generateButton = screen.getByRole('button', { name: /begin/i })
    fireEvent.click(generateButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'wellness',
        expect.objectContaining({
          kind: 'sleep',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })

  it('allows adding preferences', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<WellnessPage />)
    
    const sleepOption = screen.getByText('Sleep Story')
    fireEvent.click(sleepOption)
    
    const preferencesInput = screen.getByPlaceholderText(/e\.g\. A story about the ocean/i)
    fireEvent.change(preferencesInput, { target: { value: 'A story about the ocean' } })
    
    const generateButton = screen.getByRole('button', { name: /begin/i })
    fireEvent.click(generateButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'wellness',
        expect.objectContaining({
          preferences: 'A story about the ocean',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })
})
