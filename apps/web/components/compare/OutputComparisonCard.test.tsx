import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: mockAddToast }),
}))
vi.mock('@/lib/generate-controller', () => ({
  generateController: { generate: vi.fn() },
}))
vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual<typeof import('@sloughgpt/strui')>('@sloughgpt/strui')
  return {
    ...actual,
    Chip: ({ label, selected, onClick }: any) => (
      <button data-testid={`chip-${label}`} data-selected={selected} onClick={onClick}>{label}</button>
    ),
    Spinner: ({ size, className }: any) => <span data-testid="spinner" className={className} />,
  }
})

import OutputComparisonCard from './OutputComparisonCard'
import { generateController } from '@/lib/generate-controller'

const models = [
  { id: 'gpt2', name: 'GPT-2', loaded: true },
  { id: 'qwen', name: 'Qwen', loaded: true },
]

describe('OutputComparisonCard', () => {
  afterEach(cleanup)
  beforeEach(() => vi.clearAllMocks())

  it('renders card title', () => {
    render(<OutputComparisonCard models={models} />)
    expect(screen.getByText('Output Comparison')).toBeDefined()
  })

  it('renders comparison prompt textarea', () => {
    render(<OutputComparisonCard models={models} />)
    expect(screen.getByPlaceholderText(/Enter a prompt to compare/)).toBeDefined()
  })

  it('renders model chips', () => {
    render(<OutputComparisonCard models={models} />)
    expect(screen.getByTestId('chip-GPT-2')).toBeDefined()
    expect(screen.getByTestId('chip-Qwen')).toBeDefined()
  })

  it('toggles model selection on chip click', () => {
    render(<OutputComparisonCard models={models} />)
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    expect(screen.getByTestId('chip-GPT-2').getAttribute('data-selected')).toBe('true')
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    expect(screen.getByTestId('chip-GPT-2').getAttribute('data-selected')).toBe('false')
  })

  it('disables Compare when no prompt', () => {
    render(<OutputComparisonCard models={models} />)
    const btn = screen.getByText('Compare').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('enables Compare when prompt and model selected', async () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: 'hi', tokens_generated: 1 })
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'hello' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    const btn = screen.getByText('Compare').closest('button')!
    expect(btn.disabled).toBe(false)
  })

  it('runs comparison and displays results', async () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: 'response from model', tokens_generated: 5 })
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'test prompt' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => expect(screen.getByText('response from model')).toBeDefined())
    expect(screen.getByText('5 tok')).toBeDefined()
    expect(screen.getByText(/s$/)).toBeDefined()
  })

  it('shows error badge on generate failure', async () => {
    vi.mocked(generateController.generate).mockRejectedValue(new Error('timeout'))
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'fail' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => expect(screen.getByText('Error')).toBeDefined())
    expect(screen.getByText('timeout')).toBeDefined()
  })

  it('shows Clear button when results exist', async () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: 'ok', tokens_generated: 1 })
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'x' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => expect(screen.getByText('Clear')).toBeDefined())
  })

  it('Clear button resets results and prompt', async () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: 'ok', tokens_generated: 1 })
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'x' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => expect(screen.getByText('Clear')).toBeDefined())
    fireEvent.click(screen.getByText('Clear'))
    expect(screen.queryByText('ok')).toBeNull()
    expect((screen.getByPlaceholderText(/Enter a prompt to compare/) as HTMLTextAreaElement).value).toBe('')
  })

  it('truncates long output and shows expand button', async () => {
    const longText = 'a'.repeat(400)
    vi.mocked(generateController.generate).mockResolvedValue({ text: longText, tokens_generated: 100 })
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'long' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => expect(screen.getByText(/Show all \(400 chars\)/)).toBeDefined())
    expect(screen.getByText('a'.repeat(300) + '…')).toBeDefined()
  })

  it('copies result to clipboard on copy button click', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.assign(navigator, { clipboard: { writeText } })
    vi.mocked(generateController.generate).mockResolvedValue({ text: 'copy me', tokens_generated: 1 })
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'copy' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => expect(screen.getByLabelText('Copy response')).toBeDefined())
    fireEvent.click(screen.getByLabelText('Copy response'))
    expect(writeText).toHaveBeenCalledWith('copy me')
    expect(mockAddToast).toHaveBeenCalledWith('Copied to clipboard', 'success')
  })

  it('shows querying count while loading', async () => {
    let resolveGen: any
    vi.mocked(generateController.generate).mockImplementation(() => new Promise(r => { resolveGen = r }))
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'wait' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => expect(screen.getByText(/Querying 1 model/)).toBeDefined())
    resolveGen({ text: 'done', tokens_generated: 1 })
  })

  it('handles multiple model results in parallel', async () => {
    vi.mocked(generateController.generate).mockImplementation(async ({ model }) => ({
      text: `output-${model}`, tokens_generated: 1,
    }))
    render(<OutputComparisonCard models={models} />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to compare/), { target: { value: 'multi' } })
    fireEvent.click(screen.getByTestId('chip-GPT-2'))
    fireEvent.click(screen.getByTestId('chip-Qwen'))
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => {
      expect(screen.getByText('output-gpt2')).toBeDefined()
      expect(screen.getByText('output-qwen')).toBeDefined()
    })
  })
})
