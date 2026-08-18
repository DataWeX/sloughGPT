import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    list: vi.fn().mockResolvedValue({
      souls: [
        { name: 'Friendly', description: 'A warm companion', traits: ['warmth', 'humor'], personality: { warmth: 0.9, humor: 0.7, precision: 0.5, logic: 0.4, creativity: 0.6 }, lineage: 'gpt2', size_mb: 0.5, born_at: '2025-01-10T00:00:00Z', epochs_trained: 5, final_val_loss: 0.42 },
        { name: 'Analyst', description: 'Logical and precise', traits: ['precision', 'logic'], personality: { precision: 0.85, logic: 0.8, warmth: 0.3, humor: 0.2, creativity: 0.4 }, lineage: 'gpt2', size_mb: 0.6 },
        { name: 'Creative', description: 'Imaginative and playful', traits: ['creativity', 'humor'], personality: { creativity: 0.95, humor: 0.6, warmth: 0.5, precision: 0.3, logic: 0.4 }, lineage: 'gpt2-medium' },
      ],
      current_soul: 'Friendly',
    }),
    switch: vi.fn().mockResolvedValue(undefined),
    loadCheckpoint: vi.fn().mockResolvedValue({ status: 'loaded' }),
    getTraitWeights: vi.fn().mockResolvedValue({ personality: { warmth: 0.8 }, cognition: { reasoning: 0.7 }, emotion: { empathy: 0.5 } }),
    listWeightSnapshots: vi.fn().mockResolvedValue([{ name: 'baseline', saved_at: '2025-01-15T10:00:00Z' }]),
    saveWeightSnapshot: vi.fn().mockResolvedValue('/path'),
    loadWeightSnapshot: vi.fn().mockResolvedValue(3),
    deleteWeightSnapshot: vi.fn().mockResolvedValue(true),
    getModes: vi.fn().mockResolvedValue({ personality: { label: 'warm', confidence: 0.85 }, memory: { label: 'standard', confidence: 0.7 }, style: { label: 'formal', confidence: 0.6 }, task: { label: 'analytical', confidence: 0.75 } }),
    saveTraitWeights: vi.fn().mockResolvedValue({ status: 'ok' }),
    deleteCheckpoint: vi.fn().mockResolvedValue({ status: 'deleted' }),
    getSoul: vi.fn().mockResolvedValue({ name: 'Friendly', description: 'A warm companion' }),
    getStats: vi.fn().mockResolvedValue({ total_souls: 3, current_soul: 'Friendly' }),
    checkpointInfo: vi.fn().mockResolvedValue({ name: 'cp-warm-v2', loss: 0.45 }),
    downloadCheckpoint: vi.fn().mockResolvedValue(new Blob()),
    listCheckpoints: vi.fn().mockResolvedValue({
      checkpoints: [
        { name: 'cp-warm-v2', soul: 'Friendly', loss: 0.45, verdict: 'improved', size_mb: 1.2, perplexity_delta: -0.12, bleu_delta: 0.08 },
        { name: 'cp-base-v1', soul: 'Analyst', loss: 0.62, verdict: 'neutral', size_mb: 0.8 },
      ],
    }),
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: (s: { addToast: (...a: unknown[]) => void }) => unknown) => sel({ addToast: vi.fn() }),
}))
vi.mock('@/components/souls/SoulPersonalityCard', () => ({
  SoulPersonalityCard: () => <div data-testid="soul-personality-card" />,
}))

import SoulsPage from './page'
import { soulsController } from '@/lib/souls-controller'
const sc = vi.mocked(soulsController)

const DEFAULT_SOULS = {
  souls: [
    { name: 'Friendly', description: 'A warm companion', traits: ['warmth', 'humor'], personality: { warmth: 0.9, humor: 0.7 }, lineage: 'gpt2', size_mb: 0.5, born_at: '2025-01-10T00:00:00Z', epochs_trained: 5, final_val_loss: 0.42 },
    { name: 'Analyst', description: 'Logical and precise', traits: ['precision', 'logic'], personality: { precision: 0.85, logic: 0.8 }, lineage: 'gpt2', size_mb: 0.6 },
    { name: 'Creative', description: 'Imaginative and playful', traits: ['creativity', 'humor'], personality: { creativity: 0.95, humor: 0.6 }, lineage: 'gpt2-medium' },
  ],
  current_soul: 'Friendly',
}
const DEFAULT_CHECKPOINTS = {
  checkpoints: [
    { name: 'cp-warm-v2', soul: 'Friendly', loss: 0.45, verdict: 'improved', size_mb: 1.2, perplexity_delta: -0.12, bleu_delta: 0.08 },
    { name: 'cp-base-v1', soul: 'Analyst', loss: 0.62, verdict: 'neutral', size_mb: 0.8 },
  ],
}

async function clickTab(name: string) {
  await waitFor(() => {
    expect(screen.getAllByRole('tab').some(b => b.textContent?.trim().toLowerCase() === name.toLowerCase())).toBeTruthy()
  })
  fireEvent.click(screen.getAllByRole('tab').find(b => b.textContent?.trim().toLowerCase() === name.toLowerCase())!)
}

describe('SoulsPage', () => {
  beforeEach(() => {
    sc.list.mockResolvedValue(DEFAULT_SOULS)
    sc.listCheckpoints.mockResolvedValue(DEFAULT_CHECKPOINTS)
    sc.listWeightSnapshots.mockResolvedValue([{ name: 'baseline', saved_at: '2025-01-15T10:00:00Z' }])
    sc.getModes.mockResolvedValue({ personality: { label: 'warm', confidence: 0.85 }, memory: { label: 'standard', confidence: 0.7 }, style: { label: 'formal', confidence: 0.6 }, task: { label: 'analytical', confidence: 0.75 } })
    sc.getTraitWeights.mockResolvedValue({ personality: { warmth: 0.8 }, cognition: { reasoning: 0.7 }, emotion: { empathy: 0.5 } })
  })

  it('renders page header', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Souls')).toBeTruthy() })
  })

  it('shows KPI stats', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Personalities').length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('3')).toBeTruthy()
    })
  })

  it('shows all soul names', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('Analyst')).toBeTruthy()
      expect(screen.getByText('Creative')).toBeTruthy()
    })
  })

  it('shows soul metadata', async () => {
    render(<SoulsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('gpt2').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('0.5 MB').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('5 epochs').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('val 0.420').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows trait chips', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('warmth').length).toBeGreaterThanOrEqual(1) })
  })

  it('calls switch on button click', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Switch').length).toBeGreaterThanOrEqual(1) })
    fireEvent.click(screen.getAllByText('Switch')[0])
    await waitFor(() => { expect(sc.switch).toHaveBeenCalled() })
  })

  it('shows personality card', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByTestId('soul-personality-card').length).toBeGreaterThanOrEqual(1) })
  })

  it('filters souls by search', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    fireEvent.change(screen.getAllByPlaceholderText('Search souls...')[0], { target: { value: 'Analyst' } })
    await waitFor(() => {
      expect(screen.getByText('Analyst')).toBeTruthy()
      expect(screen.queryByText('Creative')).toBeNull()
    })
  })

  it('shows empty search state', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText('Search souls...').length).toBeGreaterThanOrEqual(1) })
    fireEvent.change(screen.getAllByPlaceholderText('Search souls...')[0], { target: { value: 'zzz' } })
    await waitFor(() => { expect(screen.getByText('No personalities match your search.')).toBeTruthy() })
  })

  it('opens detail dialog with full metadata', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    fireEvent.click(screen.getAllByText('Friendly')[0].closest('[class*="cursor-pointer"]')!)
    await waitFor(() => {
      expect(screen.getByText('A warm companion')).toBeTruthy()
      expect(screen.getByText('Training Info')).toBeTruthy()
      expect(screen.getByText('Personality')).toBeTruthy()
      expect(screen.getByText('Traits')).toBeTruthy()
    })
  })

  it('closes detail dialog on Close', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    fireEvent.click(screen.getAllByText('Friendly')[0].closest('[class*="cursor-pointer"]')!)
    await waitFor(() => { expect(screen.getAllByText('Close').length).toBeGreaterThanOrEqual(1) })
    fireEvent.click(screen.getAllByText('Close')[0])
    await waitFor(() => { expect(screen.queryByText('A warm companion')).toBeNull() })
  })

  it('shows checkpoints with verdict and loss', async () => {
    render(<SoulsPage />)
    await clickTab('checkpoints')
    await waitFor(() => {
      expect(screen.getByText('cp-warm-v2')).toBeTruthy()
      expect(screen.getAllByText('Improved').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('loss 0.450').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows perplexity and BLEU deltas', async () => {
    render(<SoulsPage />)
    await clickTab('checkpoints')
    await waitFor(() => {
      expect(screen.getAllByText('PPL -0.120').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('BLEU +0.080').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('calls loadCheckpoint', async () => {
    render(<SoulsPage />)
    await clickTab('checkpoints')
    await waitFor(() => { expect(screen.getAllByText('Load').length).toBeGreaterThanOrEqual(1) })
    fireEvent.click(screen.getAllByText('Load')[0])
    await waitFor(() => { expect(sc.loadCheckpoint).toHaveBeenCalled() })
  })

  it('shows empty checkpoints', async () => {
    sc.listCheckpoints.mockResolvedValue({ checkpoints: [] })
    render(<SoulsPage />)
    await clickTab('checkpoints')
    await waitFor(() => { expect(screen.getByText('No checkpoints found.')).toBeTruthy() })
  })

  it('shows modes and trait weights', async () => {
    render(<SoulsPage />)
    await clickTab('weights')
    await waitFor(() => {
      expect(screen.getByText('Context Modes')).toBeTruthy()
      expect(screen.getByText('Trait Weights')).toBeTruthy()
      expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows save disabled initially', async () => {
    render(<SoulsPage />)
    await clickTab('weights')
    await waitFor(() => { expect(screen.getByText('Trait Weights')).toBeTruthy() })
    expect(screen.getByText('Save').closest('button')?.disabled).toBeTruthy()
  })

  it('shows snapshots', async () => {
    render(<SoulsPage />)
    await clickTab('snapshots')
    await waitFor(() => {
      expect(screen.getByText('Weight Snapshots')).toBeTruthy()
      expect(screen.getByText('baseline')).toBeTruthy()
    })
  })

  it('calls saveWeightSnapshot', async () => {
    render(<SoulsPage />)
    await clickTab('snapshots')
    await waitFor(() => { expect(screen.getByPlaceholderText('Snapshot name...')).toBeTruthy() })
    fireEvent.change(screen.getByPlaceholderText('Snapshot name...'), { target: { value: 'new-snap' } })
    fireEvent.click(screen.getByPlaceholderText('Snapshot name...').parentElement!.querySelector('button')!)
    await waitFor(() => { expect(sc.saveWeightSnapshot).toHaveBeenCalledWith('new-snap') })
  })

  it('shows analytics overview', async () => {
    render(<SoulsPage />)
    await clickTab('analytics')
    await waitFor(() => {
      expect(screen.getByText('Soul Overview')).toBeTruthy()
      expect(screen.getByText('Personality Comparison')).toBeTruthy()
      expect(screen.getByText('Checkpoint Summary')).toBeTruthy()
    })
  })

  it('opens register dialog', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getByText('Register')).toBeTruthy() })
    fireEvent.click(screen.getByText('Register'))
    await waitFor(() => {
      expect(screen.getByText('Register Soul')).toBeTruthy()
      expect(screen.getByPlaceholderText('/absolute/path/to/soul.soul')).toBeTruthy()
    })
    fireEvent.click(screen.getAllByText('Cancel')[0])
    await waitFor(() => { expect(screen.queryByText('Register Soul')).toBeNull() })
  })
})
