import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/generate-controller', () => ({
  generateController: { generate: vi.fn() },
}))
vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual<typeof import('@sloughgpt/strui')>('@sloughgpt/strui')
  return {
    ...actual,
    Slider: ({ label, value, onValueChange, min, max, step }: any) => (
      <input
        data-testid={`slider-${label}`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value?.[0] ?? 0}
        onChange={e => onValueChange?.([parseFloat(e.target.value)])}
        aria-label={label}
      />
    ),
  }
})

import ModelPlaygroundCard from './ModelPlaygroundCard'
import { generateController } from '@/lib/generate-controller'

describe('ModelPlaygroundCard', () => {
  afterEach(cleanup)
  beforeEach(() => vi.clearAllMocks())

  it('renders null when no activeRuntimeId', () => {
    const { container } = render(<ModelPlaygroundCard activeRuntimeId={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders card title when activeRuntimeId provided', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    expect(screen.getByText('Model Playground')).toBeDefined()
  })

  it('renders textarea for prompt input', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    expect(screen.getByPlaceholderText(/Enter a prompt to test/)).toBeDefined()
  })

  it('renders temperature and max tokens sliders', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    expect(screen.getByTestId('slider-Temperature')).toBeDefined()
    expect(screen.getByTestId('slider-Max tokens')).toBeDefined()
  })

  it('renders Generate button', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    expect(screen.getByText('Generate')).toBeDefined()
  })

  it('disables Generate when prompt is empty', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    const btn = screen.getByText('Generate').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('enables Generate when prompt is entered', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'hi' } })
    const btn = screen.getByText('Generate').closest('button')!
    expect(btn.disabled).toBe(false)
  })

  it('calls generate on Generate click', async () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: 'hello output', tokens_generated: 2 })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'hi' } })
    fireEvent.click(screen.getByText('Generate'))
    expect(vi.mocked(generateController.generate)).toHaveBeenCalledWith(expect.objectContaining({ prompt: 'hi' }))
  })

  it('displays generated output', async () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: 'model response', tokens_generated: 3 })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'test' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getByText('model response')).toBeDefined())
  })

  it('shows error message on generate failure', async () => {
    vi.mocked(generateController.generate).mockRejectedValue(new Error('OOM'))
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'fail' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getByText('Error: OOM')).toBeDefined())
  })

  it('shows "No output" when result.text is empty', async () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: '', tokens_generated: 0 })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'empty' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getByText('No output')).toBeDefined())
  })

  it('Clear button resets prompt and output', async () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: 'done', tokens_generated: 1 })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'x' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getByText('done')).toBeDefined())
    fireEvent.click(screen.getByText('Clear'))
    expect((screen.getByPlaceholderText(/Enter a prompt to test/) as HTMLTextAreaElement).value).toBe('')
    expect(screen.queryByText('done')).toBeNull()
  })

  it('shows Generating... while request is pending', async () => {
    let resolve: any
    vi.mocked(generateController.generate).mockImplementation(() => new Promise(r => { resolve = r }))
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'wait' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getAllByText('Generating...').length).toBeGreaterThanOrEqual(1))
    resolve({ text: 'ok', tokens_generated: 1 })
  })

  it('applies temperature to generate call', () => {
    vi.mocked(generateController.generate).mockResolvedValue({ text: '', tokens_generated: 0 })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 't' } })
    fireEvent.click(screen.getByText('Generate'))
    expect(vi.mocked(generateController.generate)).toHaveBeenCalledWith(
      expect.objectContaining({ temperature: 0.7, max_new_tokens: 100 })
    )
  })
})
