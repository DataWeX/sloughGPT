import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/decide',
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

import DecidePage from './page'
import { generateTool } from '@/lib/tools-controller'

describe('DecidePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<DecidePage />)
    expect(screen.getByRole('heading', { name: 'Help Me Decide' })).toBeDefined()
  })

  it('has input fields for question and options', () => {
    render(<DecidePage />)
    expect(screen.getByPlaceholderText(/e\.g\. Should I take the job/i)).toBeDefined()
    expect(screen.getByPlaceholderText('First option')).toBeDefined()
    expect(screen.getByPlaceholderText('Second option')).toBeDefined()
  })

  it('has a decide button', () => {
    render(<DecidePage />)
    expect(screen.getByRole('button', { name: /decide/i })).toBeDefined()
  })

  it('shows toast when submitting empty form', async () => {
    render(<DecidePage />)

    const questionInput = screen.getByPlaceholderText(/e\.g\. Should I take the job/i)
    fireEvent.change(questionInput, { target: { value: 'Test' } })

    const decideButton = screen.getByRole('button', { name: /decide/i })
    expect(decideButton).toBeDisabled()

    fireEvent.change(questionInput, { target: { value: '' } })
    await waitFor(() => {
      expect(decideButton).toBeDisabled()
    })
  })

  it('calls generateTool when form is filled', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<DecidePage />)
    
    const questionInput = screen.getByPlaceholderText(/e\.g\. Should I take the job/i)
    fireEvent.change(questionInput, { target: { value: 'Test question' } })
    
    const optionAInput = screen.getByPlaceholderText('First option')
    fireEvent.change(optionAInput, { target: { value: 'Option A' } })
    
    const optionBInput = screen.getByPlaceholderText('Second option')
    fireEvent.change(optionBInput, { target: { value: 'Option B' } })
    
    const decideButton = screen.getByRole('button', { name: /decide/i })
    fireEvent.click(decideButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'decide',
        expect.objectContaining({
          question: 'Test question',
          option_a: 'Option A',
          option_b: 'Option B',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })
})
