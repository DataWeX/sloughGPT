import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { ModelPicker } from './model-picker'

const OPTIONS = [
  { id: 'gpt-4o', label: 'GPT-4o', badge: 'fast' },
  { id: 'claude', label: 'Claude', disabled: true },
  { id: 'mistral', label: 'Mistral' },
]

afterEach(() => {
  cleanup()
})

describe('ModelPicker', () => {
  it('renders the current option label on the trigger', () => {
    render(<ModelPicker value="gpt-4o" options={OPTIONS} onChange={() => {}} />)
    expect(screen.getByRole('button').textContent).toContain('GPT-4o')
  })

  it('falls back to the raw value when no option matches', () => {
    render(<ModelPicker value="unknown-id" options={OPTIONS} onChange={() => {}} />)
    expect(screen.getByRole('button').textContent).toContain('unknown-id')
  })

  it('opens a menu listing every option label and badge', () => {
    render(<ModelPicker value="gpt-4o" options={OPTIONS} onChange={() => {}} />)
    fireEvent.click(screen.getByRole('button'))
    expect(screen.getByText('Mistral')).toBeTruthy()
    expect(screen.getByText('Claude')).toBeTruthy()
    expect(screen.getByText('fast')).toBeTruthy()
  })

  it('marks disabled options as disabled', () => {
    render(<ModelPicker value="gpt-4o" options={OPTIONS} onChange={() => {}} />)
    fireEvent.click(screen.getByRole('button'))
    const item = screen.getByText('Claude').closest('[role="menuitem"]')
    expect(item).not.toBeNull()
    expect(item?.getAttribute('aria-disabled')).toBe('true')
  })

  it('closes the menu when an option is clicked', () => {
    const onChange = vi.fn()
    render(<ModelPicker value="gpt-4o" options={OPTIONS} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button'))
    expect(screen.getByRole('menu')).toBeTruthy()
    fireEvent.click(screen.getByText('Mistral'))
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('calls onChange with the option id on click', () => {
    const onChange = vi.fn()
    render(<ModelPicker value="gpt-4o" options={OPTIONS} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button'))
    fireEvent.click(screen.getByText('Mistral'))
    expect(onChange).toHaveBeenCalledWith('mistral')
  })

  it('does not call onChange for disabled options', () => {
    const onChange = vi.fn()
    render(<ModelPicker value="gpt-4o" options={OPTIONS} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button'))
    fireEvent.click(screen.getByText('Claude'))
    expect(onChange).not.toHaveBeenCalled()
  })

  it('disables the trigger button when disabled', () => {
    render(<ModelPicker value="gpt-4o" options={OPTIONS} onChange={() => {}} disabled />)
    const button = screen.getByRole('button') as HTMLButtonElement
    expect(button.disabled).toBe(true)
  })

  it('passes className to the trigger button', () => {
    render(<ModelPicker value="gpt-4o" options={OPTIONS} onChange={() => {}} className="picker-custom" />)
    expect(screen.getByRole('button').classList.contains('picker-custom')).toBe(true)
  })
})
