import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/brainstorm',
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

import BrainstormPage from './page'
import { generateTool } from '@/lib/tools-controller'

describe('BrainstormPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<BrainstormPage />)
    expect(screen.getByText('Brainstorm')).toBeDefined()
  })

  it('shows suggestion buttons when no messages', () => {
    render(<BrainstormPage />)
    expect(screen.getByText('Name ideas')).toBeDefined()
    expect(screen.getByText('Weekend plans')).toBeDefined()
    expect(screen.getByText('Gift ideas')).toBeDefined()
  })

  it('has an input field for typing', () => {
    render(<BrainstormPage />)
    expect(screen.getByPlaceholderText(/type your thought/i)).toBeDefined()
  })

  it('has a send button', () => {
    render(<BrainstormPage />)
    expect(screen.getByRole('button', { name: /send/i })).toBeDefined()
  })

  it('calls generateTool when sending a message', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<BrainstormPage />)
    
    const input = screen.getByPlaceholderText(/type your thought/i)
    fireEvent.change(input, { target: { value: 'Test idea' } })
    
    const sendButton = screen.getByRole('button', { name: /send/i })
    fireEvent.click(sendButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalled()
    })
  })

  it('shows user message after sending', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<BrainstormPage />)
    
    const input = screen.getByPlaceholderText(/type your thought/i)
    fireEvent.change(input, { target: { value: 'Test idea' } })
    
    const sendButton = screen.getByRole('button', { name: /send/i })
    fireEvent.click(sendButton)

    await waitFor(() => {
      expect(screen.getByText('Test idea')).toBeDefined()
    })
  })
})
