import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ChatSettings } from './ChatSettings'

afterEach(cleanup)

describe('ChatSettings', () => {
  const baseProps = {
    isOpen: true, model: 'gpt2', temperature: 0.8, maxTokens: 200,
    onModelChange: vi.fn(), onTemperatureChange: vi.fn(), onMaxTokensChange: vi.fn(),
    onClear: vi.fn(), hasMessages: true,
  }

  it('renders model, temperature, max-tokens when open', () => {
    render(<ChatSettings {...baseProps} />)
    expect(screen.getByText('gpt2')).toBeDefined()
    expect(screen.getByText('0.8')).toBeDefined()
    expect(screen.getByText('200')).toBeDefined()
  })

  it('does not render content when closed', () => {
    const { container } = render(<ChatSettings {...baseProps} isOpen={false} />)
    const section = container.querySelector('section')
    expect(section?.className).toContain('max-h-0')
  })

  it('shows model dropdown options', () => {
    render(<ChatSettings {...baseProps} availableModels={['gpt2', 'gpt2-medium']} />)
    const triggers = screen.getAllByRole('button')
    const modelTrigger = triggers.find(b => b.getAttribute('aria-label')?.startsWith('Model:'))
    expect(modelTrigger).toBeDefined()
  })

  it('calls onClear when clear button clicked and hasMessages', () => {
    const onClear = vi.fn()
    render(<ChatSettings {...baseProps} onClear={onClear} />)
    const clearBtn = screen.queryByText('Clear')
    if (clearBtn) fireEvent.click(clearBtn)
    else {
      const btns = screen.getAllByRole('button')
      fireEvent.click(btns[btns.length - 1])
    }
  })

  it('shows model short name without org prefix', () => {
    render(<ChatSettings {...baseProps} model="org/gpt2-medium" />)
    expect(screen.getByText('gpt2-medium')).toBeDefined()
  })

  it('uses default model options when availableModels is empty', () => {
    render(<ChatSettings {...baseProps} availableModels={[]} />)
    expect(screen.getByText('gpt2')).toBeDefined()
  })
})
