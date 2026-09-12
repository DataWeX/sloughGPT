import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

// ── strui mock ──
vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, disabled, className, variant, size }: any) => (
      <button onClick={onClick} disabled={disabled} className={className} data-variant={variant} data-size={size}>{children}</button>
    ),
    Input: ({ value, onChange, type, placeholder, ...props }: any) => (
      <input value={value} onChange={onChange} type={type} placeholder={placeholder} {...props} />
    ),
    Textarea: ({ value, onChange, placeholder, rows }: any) => (
      <textarea value={value} onChange={onChange} placeholder={placeholder} rows={rows} />
    ),
  }
})

// ── controller mock ──
const { mockGenerateModelCard } = vi.hoisted(() => ({
  mockGenerateModelCard: vi.fn(),
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    generateModelCard: mockGenerateModelCard,
  },
}))

// ── lucide-react mock ──
vi.mock('lucide-react', () => {
  const iconMock = (name: string) => {
    const C = () => <span data-testid={`icon-${name}`}>{name}</span>
    C.displayName = `Icon${name}`
    return C
  }
  return {
    FileText: iconMock('FileText'),
    Copy: iconMock('Copy'),
    Download: iconMock('Download'),
    Sparkles: iconMock('Sparkles'),
  }
})

// ── PageContainer & AppRouteHeader mocks ──
vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title }: any) => <div data-testid="page-container" data-title={title}>{children}</div>,
}))
vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left }: any) => <div data-testid="app-route-header">{left}</div>,
  AppRouteHeaderLead: ({ title }: any) => <span>{title}</span>,
}))

import Page from './page'

function fillName(value: string) {
  const input = screen.getByPlaceholderText('my-fine-tuned-model')
  fireEvent.change(input, { target: { value } })
}

afterEach(() => { cleanup() })

beforeEach(() => {
  vi.clearAllMocks()
  mockGenerateModelCard.mockResolvedValue({
    card: { model_type: 'llama' },
    markdown: '# My Model\n\nThis is a generated model card.',
  })
})

describe('ModelCardPage', () => {
  it('renders without crashing', () => {
    render(<Page />)
    expect(screen.getByTestId('page-container')).toBeTruthy()
  })

  it('renders page title', () => {
    render(<Page />)
    expect(screen.getByTestId('page-container').getAttribute('data-title')).toBe('Model Card Generator')
    expect(screen.getByText('Model Card Generator')).toBeTruthy()
  })

  it('renders form section with inputs', () => {
    render(<Page />)
    expect(screen.getByText('Model Details')).toBeTruthy()
    expect(screen.getByPlaceholderText('my-fine-tuned-model')).toBeTruthy()
    expect(screen.getByPlaceholderText('gpt2')).toBeTruthy()
    expect(screen.getByPlaceholderText('A fine-tuned model for...')).toBeTruthy()
    expect(screen.getByPlaceholderText('dataset-name')).toBeTruthy()
    expect(screen.getByPlaceholderText('finetune')).toBeTruthy()
  })

  it('renders preview section with empty state', () => {
    render(<Page />)
    expect(screen.getByText('Preview')).toBeTruthy()
    expect(screen.getByText('Fill in the form and click Generate to create a model card.')).toBeTruthy()
  })

  it('renders Generate Model Card button', () => {
    render(<Page />)
    expect(screen.getByText('Generate Model Card')).toBeTruthy()
  })

  it('disables Generate button when name is empty', () => {
    render(<Page />)
    const btn = screen.getByText('Generate Model Card')
    expect(btn.closest('button')?.disabled).toBe(true)
  })

  it('enables Generate button when name is filled', () => {
    render(<Page />)
    fillName('my-model')
    const btn = screen.getByText('Generate Model Card')
    expect(btn.closest('button')?.disabled).toBe(false)
  })

  it('shows loading state while generating', async () => {
    mockGenerateModelCard.mockReturnValue(new Promise(() => {}))
    render(<Page />)
    fillName('my-model')
    await act(async () => {
      screen.getByText('Generate Model Card').click()
    })
    expect(screen.getByText('Generating...')).toBeTruthy()
    expect(screen.getByText('Generating...').closest('button')?.disabled).toBe(true)
  })

  it('calls settingsController.generateModelCard on generate', async () => {
    render(<Page />)
    fillName('my-model')
    await act(async () => {
      screen.getByText('Generate Model Card').click()
    })
    expect(mockGenerateModelCard).toHaveBeenCalledTimes(1)
    expect(mockGenerateModelCard).toHaveBeenCalledWith(
      'my-model',
      expect.objectContaining({
        base_model: 'gpt2',
        training_method: 'finetune',
      }),
    )
  })

  it('shows generated markdown in preview after successful generate', async () => {
    render(<Page />)
    fillName('my-model')
    await act(async () => {
      screen.getByText('Generate Model Card').click()
    })
    await waitFor(() => {
      expect(screen.getByText(/My Model/)).toBeTruthy()
    })
    expect(screen.getAllByText('Copy').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Download .md').length).toBeGreaterThanOrEqual(1)
  })

  it('shows error state when generate fails', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    mockGenerateModelCard.mockRejectedValue(new Error('API error'))
    render(<Page />)
    fillName('my-model')
    await act(async () => {
      screen.getByText('Generate Model Card').click()
    })
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to generate model card:', expect.any(Error))
    })
    consoleSpy.mockRestore()
  })

  it('hides empty state and shows preview content after generation', async () => {
    render(<Page />)
    expect(screen.getByText('Fill in the form and click Generate to create a model card.')).toBeTruthy()
    fillName('my-model')
    await act(async () => {
      screen.getByText('Generate Model Card').click()
    })
    await waitFor(() => {
      expect(screen.getByText(/My Model/)).toBeTruthy()
    })
    expect(screen.queryByText('Fill in the form and click Generate to create a model card.')).toBeNull()
  })

  it('does not generate when name is empty', async () => {
    render(<Page />)
    await act(async () => {
      screen.getByText('Generate Model Card').click()
    })
    expect(mockGenerateModelCard).not.toHaveBeenCalled()
  })
})
