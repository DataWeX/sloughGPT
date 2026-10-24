import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    Button: ({ children, onClick, disabled, className }: any) => (
      <button onClick={onClick} disabled={disabled} className={className}>{children}</button>
    ),
    Dialog: ({ children }: any) => <div>{children}</div>,
    DialogContent: ({ children }: any) => <div>{children}</div>,
    DialogHeader: ({ children }: any) => <div>{children}</div>,
    DialogTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, ...props }: any) => <input value={value} onChange={onChange} {...props} />,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
  }
})

vi.mock('lucide-react', () => ({
  FileText: () => <span />,
  Copy: () => <span />,
  Download: () => <span />,
  Sparkles: () => <span />,
}))

const { mockGenerate } = vi.hoisted(() => ({ mockGenerate: vi.fn() }))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: { generateModelCard: (...a: any[]) => mockGenerate(...a) },
}))

vi.mock('@/lib/download-utils', () => ({ downloadMarkdown: vi.fn() }))

import { ModelCardDialog } from './ModelCardDialog'

afterEach(cleanup)
beforeEach(() => {
  vi.clearAllMocks()
  mockGenerate.mockResolvedValue({ markdown: '# Card' })
})

describe('ModelCardDialog', () => {
  it('disables Generate without a name and calls generate with one', async () => {
    render(<ModelCardDialog open onClose={() => {}} />)
    const generateBtn = screen.getByRole('button', { name: 'Generate Model Card' })
    expect(generateBtn).toBeDisabled()
    await act(async () => {
      fireEvent.change(screen.getByLabelText('Model name'), { target: { value: 'my-model' } })
    })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Generate Model Card' })) })
    expect(mockGenerate).toHaveBeenCalledWith('my-model', expect.objectContaining({ base_model: 'gpt2' }))
    expect(await screen.findByText('# Card')).toBeTruthy()
  })

  it('prefills name/model/dataset from run context', () => {
    render(<ModelCardDialog open onClose={() => {}} initialName="run-1" initialModel="llama" initialDataset="ds" />)
    expect((screen.getByLabelText('Model name') as HTMLInputElement).value).toBe('run-1')
    expect((screen.getByLabelText('Base model') as HTMLInputElement).value).toBe('llama')
  })
})
