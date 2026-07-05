import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('./ModelDropdown', () => ({
  ModelDropdown: ({ variant }: any) => <div data-testid="model-dropdown" data-variant={variant} />,
}))

vi.mock('@/components/ui', () => ({
  IconCheck: () => <span data-testid="icon-check">check</span>,
}))

import { CheckpointsTab } from './CheckpointsTab'

describe('CheckpointsTab', () => {
  const onLoadCheckpoint = vi.fn()
  const onSelectModel = vi.fn()
  const onSwitchSoul = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders personality section when souls provided', () => {
    render(
      <CheckpointsTab
        checkpoints={[]}
        onLoadCheckpoint={onLoadCheckpoint}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[{ name: 'friendly' }, { name: 'witty' }]}
        currentSoulName="friendly"
        onSwitchSoul={onSwitchSoul}
      />
    )
    expect(screen.getByText('Personality')).toBeDefined()
    expect(screen.getByText('friendly')).toBeDefined()
    expect(screen.getByText('witty')).toBeDefined()
  })

  it('does not render personality section when souls empty', () => {
    render(
      <CheckpointsTab
        checkpoints={[]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.queryByText('Personality')).toBeNull()
  })

  it('highlights current soul with check icon', () => {
    render(
      <CheckpointsTab
        checkpoints={[]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[{ name: 'friendly' }, { name: 'witty' }]}
        currentSoulName="friendly"
        onSwitchSoul={onSwitchSoul}
      />
    )
    const checkIcons = screen.getAllByTestId('icon-check')
    expect(checkIcons.length).toBeGreaterThanOrEqual(1)
  })

  it('calls onSwitchSoul when soul button clicked', () => {
    render(
      <CheckpointsTab
        checkpoints={[]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[{ name: 'witty' }]}
        currentSoulName="friendly"
        onSwitchSoul={onSwitchSoul}
      />
    )
    fireEvent.click(screen.getByText('witty'))
    expect(onSwitchSoul).toHaveBeenCalledWith('witty')
  })

  it('renders ModelDropdown', () => {
    render(
      <CheckpointsTab
        checkpoints={[]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.getByTestId('model-dropdown')).toBeDefined()
  })

  it('renders Checkpoints header', () => {
    render(
      <CheckpointsTab
        checkpoints={[]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.getByText('Checkpoints')).toBeDefined()
  })

  it('shows checkpoint count', () => {
    render(
      <CheckpointsTab
        checkpoints={[{ name: 'ckpt1' }, { name: 'ckpt2' }]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.getByText('2 saved')).toBeDefined()
  })

  it('shows empty state when no checkpoints', () => {
    render(
      <CheckpointsTab
        checkpoints={[]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.getByText('No checkpoints yet')).toBeDefined()
  })

  it('renders checkpoint names', () => {
    render(
      <CheckpointsTab
        checkpoints={[{ name: 'best-model-v3' }, { name: 'after-feedback' }]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.getByText('best-model-v3')).toBeDefined()
    expect(screen.getByText('after-feedback')).toBeDefined()
  })

  it('shows check icon on loaded checkpoint', () => {
    render(
      <CheckpointsTab
        checkpoints={[{ name: 'ckpt1', is_loaded: true }, { name: 'ckpt2' }]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    const checkIcons = screen.getAllByTestId('icon-check')
    expect(checkIcons.length).toBe(1)
  })

  it('shows eval verdict badge', () => {
    render(
      <CheckpointsTab
        checkpoints={[{ name: 'ckpt1', eval_verdict: 'PASS' }, { name: 'ckpt2', eval_verdict: 'FAIL' }]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.getByText('PASS')).toBeDefined()
    expect(screen.getByText('FAIL')).toBeDefined()
  })

  it('shows loss value', () => {
    render(
      <CheckpointsTab
        checkpoints={[{ name: 'ckpt1', loss: 0.1234 }]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.getByText(/loss 0.1234/)).toBeDefined()
  })

  it('shows traits', () => {
    render(
      <CheckpointsTab
        checkpoints={[{ name: 'ckpt1', traits: ['warm', 'curious'] }]}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    expect(screen.getByText(/warm . curious/)).toBeDefined()
  })

  it('calls onLoadCheckpoint when checkpoint button clicked', () => {
    render(
      <CheckpointsTab
        checkpoints={[{ name: 'my-ckpt' }]}
        onLoadCheckpoint={onLoadCheckpoint}
        availableModels={['gpt2']}
        currentModel="gpt2"
        onSelectModel={onSelectModel}
        souls={[]}
      />
    )
    fireEvent.click(screen.getByText('my-ckpt'))
    expect(onLoadCheckpoint).toHaveBeenCalledWith('my-ckpt')
  })
})
