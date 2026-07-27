import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

import ComposableLayersCard from '@/components/models/ComposableLayersCard'

describe('ComposableLayersCard', () => {
  it('renders title', () => {
    render(<ComposableLayersCard modelsCount={0} soulsCount={0} checkpoints={[]} />)
    expect(screen.getAllByText('Composable Layers').length).toBeGreaterThanOrEqual(1)
  })

  it('renders all four layer titles', () => {
    render(<ComposableLayersCard modelsCount={0} soulsCount={0} checkpoints={[]} />)
    expect(screen.getAllByText('Base Models').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Personalities').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Adapters').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Checkpoints').length).toBeGreaterThanOrEqual(1)
  })

  it('shows modelsCount and soulsCount', () => {
    render(<ComposableLayersCard modelsCount={5} soulsCount={3} checkpoints={[]} />)
    expect(screen.getAllByText('5 available').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('3 available').length).toBeGreaterThanOrEqual(1)
  })

  it('counts checkpoints total', () => {
    const checkpoints = [
      { name: 'c1', soul: 's1', created_at: '', loss: 0, epochs: 0, dataset: '' },
      { name: 'c2', soul: '', created_at: '', loss: 0, epochs: 0, dataset: '' },
    ]
    render(<ComposableLayersCard modelsCount={0} soulsCount={0} checkpoints={checkpoints} />)
    expect(screen.getAllByText('2 available').length).toBeGreaterThanOrEqual(1)
  })

  it('counts adapters as checkpoints with soul', () => {
    const checkpoints = [
      { name: 'c1', soul: 's1', created_at: '', loss: 0, epochs: 0, dataset: '' },
      { name: 'c2', soul: 's2', created_at: '', loss: 0, epochs: 0, dataset: '' },
      { name: 'c3', soul: '', created_at: '', loss: 0, epochs: 0, dataset: '' },
    ]
    render(<ComposableLayersCard modelsCount={0} soulsCount={0} checkpoints={checkpoints} />)
    const availableTexts = screen.getAllByText(/available/)
    const counts = availableTexts.map(el => el.textContent)
    expect(counts).toContain('2 available')
  })

  it('shows layer descriptions', () => {
    render(<ComposableLayersCard modelsCount={0} soulsCount={0} checkpoints={[]} />)
    expect(screen.getAllByText(/HuggingFace model/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Soul profiles/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/LoRA\/DoRA/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Trained checkpoints/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows layer icons', () => {
    render(<ComposableLayersCard modelsCount={0} soulsCount={0} checkpoints={[]} />)
    expect(screen.getAllByText('🧠').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('🎭').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('🧩').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('📦').length).toBeGreaterThanOrEqual(1)
  })
})
