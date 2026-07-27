import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual<typeof import('@sloughgpt/strui')>('@sloughgpt/strui')
  return {
    ...actual,
    Select: ({ children, value, onValueChange, disabled }: any) => (
      <div data-testid="select" data-value={value} data-disabled={disabled}>
        {React.Children.map(children, child =>
          React.isValidElement(child)
            ? React.cloneElement(child as React.ReactElement<any>, { onValueChange })
            : child
        )}
      </div>
    ),
    SelectTrigger: ({ children, ...props }: any) => <div role="combobox" {...props}>{children}</div>,
    SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
    SelectContent: ({ children }: any) => <div>{children}</div>,
    SelectItem: ({ children, value, ...props }: any) => (
      <div role="option" data-value={value} {...props}>{children}</div>
    ),
  }
})

import PersonalitiesCard from './PersonalitiesCard'
import type { Soul, Checkpoint } from '@/lib/souls-controller'

describe('PersonalitiesCard', () => {
  afterEach(cleanup)

  const souls = [
    { name: 'friendly', description: 'Warm and approachable', traits: ['warm'], personality: { warmth: 0.8 } },
    { name: 'witty', description: 'Sharp and funny', traits: ['funny'], personality: { humor: 0.9 } },
  ] as Soul[]
  const base = {
    souls, soulsLoading: false,
    checkpoints: [], checkpointsLoading: false,
    currentSoul: null, activeCheckpoint: null,
    switchingSoul: null, onSwitch: vi.fn(),
  }

  it('renders soul names', () => {
    render(<PersonalitiesCard {...base} />)
    expect(screen.getByText('friendly')).toBeDefined()
    expect(screen.getByText('witty')).toBeDefined()
  })

  it('renders soul descriptions', () => {
    render(<PersonalitiesCard {...base} />)
    expect(screen.getByText('Warm and approachable')).toBeDefined()
    expect(screen.getByText('Sharp and funny')).toBeDefined()
  })

  it('shows loading skeleton when loading', () => {
    const { container } = render(<PersonalitiesCard {...base} soulsLoading souls={[]} />)
    expect(container.querySelector('.animate-pulse')).toBeDefined()
  })

  it('returns null when empty and not loading', () => {
    const { container } = render(<PersonalitiesCard {...base} souls={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows active badge for current soul', () => {
    render(<PersonalitiesCard {...base} currentSoul="friendly" />)
    expect(screen.getAllByText('Active').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Switch button for non-current soul without checkpoints', () => {
    render(<PersonalitiesCard {...base} currentSoul="friendly" />)
    const buttons = screen.getAllByText('Switch')
    expect(buttons.length).toBeGreaterThanOrEqual(1)
  })

  it('calls onSwitch on button click', () => {
    const onSwitch = vi.fn()
    render(<PersonalitiesCard {...base} currentSoul="friendly" onSwitch={onSwitch} />)
    const buttons = screen.getAllByText('Switch')
    fireEvent.click(buttons[0])
    expect(onSwitch).toHaveBeenCalledWith('witty')
  })

  it('renders checkpoint selector when checkpoints exist', () => {
    const checkpoints = [{ name: 'v1', soul: 'friendly', traits: {}, saved_at: '', size_mb: 0 }] as unknown as Checkpoint[]
    render(<PersonalitiesCard {...base} currentSoul="friendly" checkpoints={checkpoints} />)
    const selects = document.querySelectorAll('[role="combobox"]')
    expect(selects.length).toBeGreaterThanOrEqual(1)
  })

  it('shows checkpoint loss and date in selector', () => {
    const checkpoints = [{
      name: 'v1', soul: 'friendly', loss: 0.42, born_at: '2025-06-15T10:00:00Z',
    }] as unknown as Checkpoint[]
    render(<PersonalitiesCard {...base} currentSoul="friendly" checkpoints={checkpoints} />)
    expect(screen.getByText(/loss 0\.42/)).toBeDefined()
    expect(screen.getByText(/Jun 15/)).toBeDefined()
  })

  it('shows checkpointsLoading spinner when loading checkpoints', () => {
    const { container } = render(<PersonalitiesCard {...base} currentSoul="witty" checkpointsLoading />)
    const pulse = container.querySelector('.animate-pulse')
    expect(pulse).toBeDefined()
  })

  it('shows spinner during switching', () => {
    const { container } = render(<PersonalitiesCard {...base} currentSoul="witty" switchingSoul="witty" />)
    const spinner = container.querySelector('.animate-spin')
    expect(spinner).toBeDefined()
  })

  it('disables Switch button during switching', () => {
    const { container } = render(<PersonalitiesCard {...base} currentSoul="friendly" switchingSoul="witty" />)
    const spinner = container.querySelector('.animate-spin')
    expect(spinner).toBeDefined()
    const btn = spinner?.closest('button')
    expect(btn?.disabled).toBe(true)
  })

  it('shows final_train_loss when available', () => {
    const checkpoints = [{
      name: 'v2', soul: 'witty', final_train_loss: 0.31, born_at: '2025-07-01T00:00:00Z',
    }] as unknown as Checkpoint[]
    render(<PersonalitiesCard {...base} currentSoul="witty" checkpoints={checkpoints} />)
    expect(screen.getByText(/loss 0\.31/)).toBeDefined()
  })
})
