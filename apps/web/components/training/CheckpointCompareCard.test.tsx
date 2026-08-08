// @vitest-environment jsdom
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { CheckpointCompareCard } from './CheckpointCompareCard'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'test', soul: 'test', ...overrides }
}

describe('CheckpointCompareCard', () => {
  it('returns null when fewer than 2 checkpoints', () => {
    const { container } = render(<CheckpointCompareCard checkpoints={[mkCp()]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders heading with 2+ checkpoints', () => {
    render(<CheckpointCompareCard checkpoints={[mkCp({ name: 'a' }), mkCp({ name: 'b' })]} />)
    expect(screen.getByText('Compare checkpoints')).toBeDefined()
  })

  it('toggles expanded state', () => {
    render(<CheckpointCompareCard checkpoints={[mkCp({ name: 'a' }), mkCp({ name: 'b' })]} />)
    const btn = screen.getAllByText('Compare').find(el => el.tagName === 'BUTTON')!
    fireEvent.click(btn)
    expect(screen.getByText('Checkpoint A')).toBeDefined()
    expect(screen.getByText('Checkpoint B')).toBeDefined()
    expect(screen.getByText('Hide')).toBeDefined()
    fireEvent.click(screen.getByText('Hide'))
    expect(screen.queryByText('Checkpoint A')).toBeNull()
  })

  it('returns null for empty array', () => {
    const { container } = render(<CheckpointCompareCard checkpoints={[]} />)
    expect(container.innerHTML).toBe('')
  })
})
