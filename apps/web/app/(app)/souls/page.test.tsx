import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    list: vi.fn(),
    switch: vi.fn(),
    loadCheckpoint: vi.fn(),
    getTraitWeights: vi.fn(),
    listWeightSnapshots: vi.fn(),
    saveWeightSnapshot: vi.fn(),
    loadWeightSnapshot: vi.fn(),
    deleteWeightSnapshot: vi.fn(),
    getModes: vi.fn(),
    saveTraitWeights: vi.fn(),
    deleteCheckpoint: vi.fn(),
    getSoul: vi.fn(),
    getStats: vi.fn(),
    checkpointInfo: vi.fn(),
    downloadCheckpoint: vi.fn(),
    listCheckpoints: vi.fn(),
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: (s: { addToast: (...a: unknown[]) => void }) => unknown) => sel({ addToast: vi.fn() }),
}))
vi.mock('@/components/souls/SoulPersonalityCard', () => ({
  SoulPersonalityCard: () => <div data-testid="soul-personality-card" />,
}))

import SoulsPage from './page'
import { soulsController, type Soul, type Checkpoint } from '@/lib/souls-controller'
const sc = vi.mocked(soulsController)

const SOULS: Soul[] = [
  { name: 'Friendly', description: 'A warm companion', traits: ['warmth', 'humor'], personality: { warmth: 0.9, humor: 0.7 }, lineage: 'gpt2', size_mb: 0.5, born_at: '2025-01-10T00:00:00Z', epochs_trained: 5, final_val_loss: 0.42 },
  { name: 'Analyst', description: 'Logical and precise', traits: ['precision', 'logic'], personality: { precision: 0.85, logic: 0.8 }, lineage: 'gpt2', size_mb: 0.6 },
  { name: 'Creative', description: 'Imaginative and playful', traits: ['creativity', 'humor'], personality: { creativity: 0.95, humor: 0.6 }, lineage: 'gpt2-medium' },
]
const CHECKPOINTS: Checkpoint[] = [
  { name: 'cp-warm-v2', soul: 'Friendly', loss: 0.45, verdict: 'improved', size_mb: 1.2, perplexity_delta: -0.12, bleu_delta: 0.08 },
  { name: 'cp-base-v1', soul: 'Analyst', loss: 0.62, verdict: 'neutral', size_mb: 0.8 },
]

async function clickTab(name: string) {
  await waitFor(() => {
    expect(screen.getAllByRole('tab').some(b => b.textContent?.trim().toLowerCase() === name.toLowerCase())).toBeTruthy()
  })
  fireEvent.click(screen.getAllByRole('tab').find(b => b.textContent?.trim().toLowerCase() === name.toLowerCase())!)
}

function clickSoulCard(name: string) {
  const spans = screen.getAllByText(name)
  for (const span of spans) {
    const card = span.closest('[class*="cursor-pointer"]')
    if (card) { fireEvent.click(card); return }
  }
  throw new Error(`No clickable card found for "${name}"`)
}

function clickCheckpointRow(name: string) {
  const el = screen.getByText(name).closest('[class*="cursor-pointer"]')
  if (el) fireEvent.click(el)
}

describe('SoulsPage', () => {
  afterEach(cleanup)

  beforeEach(() => {
    sc.list.mockResolvedValue({ souls: SOULS, current_soul: 'Friendly' })
    sc.listCheckpoints.mockResolvedValue({ checkpoints: CHECKPOINTS })
    sc.listWeightSnapshots.mockResolvedValue([{ name: 'baseline', saved_at: '2025-01-15T10:00:00Z' }])
    sc.getModes.mockResolvedValue({ personality: { label: 'warm', confidence: 0.85 }, memory: { label: 'standard', confidence: 0.7 }, style: { label: 'formal', confidence: 0.6 }, task: { label: 'analytical', confidence: 0.75 } })
    sc.getTraitWeights.mockResolvedValue({ personality: { warmth: 0.8 }, cognition: { reasoning: 0.7 }, emotion: { empathy: 0.5 } })
    sc.getSoul.mockResolvedValue({ name: 'Friendly', description: 'A warm companion', personality: { warmth: 0.9 }, traits: ['warmth'] } as Soul)
    sc.checkpointInfo.mockResolvedValue(CHECKPOINTS[0])
    sc.downloadCheckpoint.mockResolvedValue(new Blob())
    sc.deleteCheckpoint.mockResolvedValue({ status: 'deleted' } as never)
    sc.deleteWeightSnapshot.mockResolvedValue(true)
    sc.loadWeightSnapshot.mockResolvedValue(3)
    sc.saveWeightSnapshot.mockResolvedValue('/path')
    sc.saveTraitWeights.mockResolvedValue({ status: 'ok' } as never)
    sc.switch.mockResolvedValue(undefined as never)
    sc.loadCheckpoint.mockResolvedValue({ status: 'loaded' } as never)
    sc.getStats.mockResolvedValue({ total_souls: 3, current_soul: 'Friendly', available_souls: ['Friendly', 'Analyst', 'Creative'] })
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
      expect(screen.getAllByText('Analyst').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Creative').length).toBeGreaterThanOrEqual(1)
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
    const user = userEvent.setup()
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    const input = screen.getAllByPlaceholderText('Search souls...')[0]
    await user.clear(input)
    await user.type(input, 'Analyst')
    await waitFor(() => {
      expect(screen.getAllByText('Analyst').length).toBeGreaterThanOrEqual(1)
      expect(screen.queryAllByText('Creative').length).toBe(0)
    })
  })

  it('shows empty search state', async () => {
    const user = userEvent.setup()
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText('Search souls...').length).toBeGreaterThanOrEqual(1) })
    const input = screen.getAllByPlaceholderText('Search souls...')[0]
    await user.clear(input)
    await user.type(input, 'zzz')
    await waitFor(() => { expect(screen.getByText('No personalities match your search.')).toBeTruthy() })
  })

  it('opens detail dialog with full metadata', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    clickSoulCard('Friendly')
    await waitFor(() => {
      expect(screen.getAllByText('A warm companion').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Training Info').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Personality').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Traits').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('closes detail dialog on Close', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    clickSoulCard('Friendly')
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

  it('opens checkpoint detail dialog', async () => {
    render(<SoulsPage />)
    await clickTab('checkpoints')
    await waitFor(() => { expect(screen.getByText('cp-warm-v2')).toBeTruthy() })
    clickCheckpointRow('cp-warm-v2')
    await waitFor(() => {
      expect(screen.getAllByText('cp-warm-v2').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1)
    })
    expect(sc.checkpointInfo).toHaveBeenCalledWith('cp-warm-v2')
  })

  it('downloads checkpoint', async () => {
    render(<SoulsPage />)
    await clickTab('checkpoints')
    await waitFor(() => { expect(screen.getAllByText('Load').length).toBeGreaterThanOrEqual(1) })
    const loadRow = screen.getByText('cp-warm-v2').closest('[class*="rounded"]')!
    const downloadBtn = loadRow.querySelectorAll('button')[1]
    fireEvent.click(downloadBtn)
    await waitFor(() => { expect(sc.downloadCheckpoint).toHaveBeenCalled() })
  })

  it('opens delete confirmation for checkpoint', async () => {
    render(<SoulsPage />)
    await clickTab('checkpoints')
    await waitFor(() => { expect(screen.getAllByText('Load').length).toBeGreaterThanOrEqual(1) })
    const trashBtns = screen.getAllByRole('button').filter(b => b.className.includes('destructive'))
    if (trashBtns.length > 0) {
      fireEvent.click(trashBtns[0])
      await waitFor(() => { expect(screen.getByText('Delete checkpoint?')).toBeTruthy() })
    }
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

  it('loads snapshot', async () => {
    render(<SoulsPage />)
    await clickTab('snapshots')
    await waitFor(() => { expect(screen.getByText('baseline')).toBeTruthy() })
    const loadBtn = screen.getAllByText('Load').find(b => b.closest('[class*="border-border"]'))
    if (loadBtn) {
      fireEvent.click(loadBtn)
      await waitFor(() => { expect(sc.loadWeightSnapshot).toHaveBeenCalledWith('baseline') })
    }
  })

  it('opens delete confirmation for snapshot', async () => {
    render(<SoulsPage />)
    await clickTab('snapshots')
    await waitFor(() => { expect(screen.getByText('baseline')).toBeTruthy() })
    const trashBtns = screen.getAllByRole('button').filter(b => b.className.includes('destructive'))
    if (trashBtns.length > 0) {
      fireEvent.click(trashBtns[0])
      await waitFor(() => { expect(screen.getByText('Delete snapshot?')).toBeTruthy() })
    }
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

  it('shows personality comparison chart with multiple souls', async () => {
    render(<SoulsPage />)
    await clickTab('analytics')
    await waitFor(() => { expect(screen.getByText('Personality Comparison')).toBeTruthy() })
    expect(screen.getByText('Warmth')).toBeTruthy()
    expect(screen.getByText('Humor')).toBeTruthy()
  })

  it('shows checkpoint summary counts', async () => {
    render(<SoulsPage />)
    await clickTab('analytics')
    await waitFor(() => { expect(screen.getByText('Checkpoint Summary')).toBeTruthy() })
    expect(screen.getByText('Total')).toBeTruthy()
    expect(screen.getByText('Improved')).toBeTruthy()
    expect(screen.getByText('Degraded')).toBeTruthy()
  })

  it('opens register dialog', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Register').length).toBeGreaterThanOrEqual(1) })
    const registerBtn = screen.getAllByRole('button', { name: /register/i })[0]
    fireEvent.click(registerBtn)
    await waitFor(() => {
      expect(screen.getByText('Register Soul')).toBeTruthy()
      expect(screen.getByPlaceholderText('/absolute/path/to/soul.soul')).toBeTruthy()
    })
    fireEvent.click(screen.getAllByText('Cancel')[0])
    await waitFor(() => { expect(screen.queryByText('Register Soul')).toBeNull() })
  })

  it('refreshes data', async () => {
    render(<SoulsPage />)
    await waitFor(() => { expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
    const refreshBtns = screen.getAllByRole('button').filter(b => b.querySelector('svg.h-4'))
    if (refreshBtns.length > 0) {
      fireEvent.click(refreshBtns[0])
      await waitFor(() => { expect(sc.list).toHaveBeenCalledTimes(2) })
    }
  })
})
