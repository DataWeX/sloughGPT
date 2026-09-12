import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/translate',
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

import TranslatePage from './page'
import { generateTool } from '@/lib/tools-controller'

describe('TranslatePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<TranslatePage />)
    expect(screen.getByRole('heading', { name: 'Translate' })).toBeDefined()
  })

  it('has a textarea for source text', () => {
    render(<TranslatePage />)
    expect(screen.getByPlaceholderText(/type or paste text to translate/i)).toBeDefined()
  })

  it('has a language selector', () => {
    render(<TranslatePage />)
    expect(screen.getByRole('combobox')).toBeDefined()
  })

  it('has a translate button', () => {
    render(<TranslatePage />)
    expect(screen.getByRole('button', { name: /translate/i })).toBeDefined()
  })

  it('disables the translate button with empty text', async () => {
    render(<TranslatePage />)
    
    const translateButton = screen.getByRole('button', { name: /translate/i })
    expect(translateButton.hasAttribute('disabled')).toBe(true)
  })

  it('calls generateTool when translating text', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<TranslatePage />)
    
    const sourceTextarea = screen.getByPlaceholderText(/type or paste text to translate/i)
    fireEvent.change(sourceTextarea, { target: { value: 'Hello world' } })
    
    const translateButton = screen.getByRole('button', { name: /translate/i })
    fireEvent.click(translateButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'translate',
        expect.objectContaining({
          text: 'Hello world',
          target_lang: 'Spanish',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })

  it('allows changing target language', async () => {
    const mockGenerateTool = vi.mocked(generateTool)
    mockGenerateTool.mockResolvedValue(undefined)

    render(<TranslatePage />)
    
    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByText('French'))
    
    const sourceTextarea = screen.getByPlaceholderText(/type or paste text to translate/i)
    fireEvent.change(sourceTextarea, { target: { value: 'Hello world' } })
    
    const translateButton = screen.getByRole('button', { name: /translate/i })
    fireEvent.click(translateButton)

    await waitFor(() => {
      expect(mockGenerateTool).toHaveBeenCalledWith(
        'translate',
        expect.objectContaining({
          target_lang: 'French',
        }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })
})
