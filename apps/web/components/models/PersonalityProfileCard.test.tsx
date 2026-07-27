import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: mockAddToast }),
}))
vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    listWeightSnapshots: vi.fn().mockResolvedValue([]),
    saveWeightSnapshot: vi.fn().mockResolvedValue('/snap'),
    loadWeightSnapshot: vi.fn().mockResolvedValue(3),
    deleteWeightSnapshot: vi.fn().mockResolvedValue(true),
  },
}))
vi.mock('@/components/souls/SoulVisualizer', () => ({
  default: ({ traitWeights, currentSoulName }: any) => (
    <div data-testid="soul-visualizer" data-name={currentSoulName}>visualizer</div>
  ),
}))
vi.mock('@/components/souls/TraitEditor', () => ({
  default: ({ traitWeights, onSave, onReset }: any) => (
    <div data-testid="trait-editor">
      <button onClick={() => onSave({ personality: { warm: 0.9 } })}>save</button>
      <button onClick={onReset}>cancel</button>
    </div>
  ),
}))

import PersonalityProfileCard from './PersonalityProfileCard'
import { soulsController } from '@/lib/souls-controller'

const defaultWeights = { personality: { warm: 0.8, humor: 0.5 } }

describe('PersonalityProfileCard', () => {
  afterEach(cleanup)
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(soulsController.listWeightSnapshots).mockResolvedValue([])
    vi.mocked(soulsController.saveWeightSnapshot).mockResolvedValue('/snap')
    vi.mocked(soulsController.loadWeightSnapshot).mockResolvedValue(3)
    vi.mocked(soulsController.deleteWeightSnapshot).mockResolvedValue(true)
  })

  it('renders null when traitWeights is null', () => {
    const { container } = render(<PersonalityProfileCard traitWeights={null} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders card title and description', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    expect(screen.getByText('Personality Profile')).toBeDefined()
    expect(screen.getByText(/Traits shape how your personality responds/)).toBeDefined()
  })

  it('shows SoulVisualizer in view mode by default', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    expect(screen.getByTestId('soul-visualizer')).toBeDefined()
    expect(screen.queryByTestId('trait-editor')).toBeNull()
  })

  it('toggles to TraitEditor when Edit clicked', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} currentSoulName="warm" onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    fireEvent.click(screen.getByText('Edit'))
    expect(screen.getByTestId('trait-editor')).toBeDefined()
    expect(screen.queryByTestId('soul-visualizer')).toBeNull()
  })

  it('toggles back to SoulVisualizer when View clicked', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    fireEvent.click(screen.getByText('Edit'))
    expect(screen.getByTestId('trait-editor')).toBeDefined()
    fireEvent.click(screen.getByText('View'))
    expect(screen.getByTestId('soul-visualizer')).toBeDefined()
  })

  it('shows snapshots section with count', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    expect(screen.getByText('Snapshots (0)')).toBeDefined()
  })

  it('shows empty snapshot hint when no snapshots', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    expect(screen.getByText(/Save weight presets to switch between personalities quickly/)).toBeDefined()
  })

  it('renders snapshot name input and Save button', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    expect(screen.getByPlaceholderText('Name this state...')).toBeDefined()
    expect(screen.getByText('Save')).toBeDefined()
  })

  it('Save button is disabled when snapshot name is empty', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    const saveBtn = screen.getByText('Save').closest('button')!
    expect(saveBtn.disabled).toBe(true)
  })

  it('saves snapshot when name is entered', async () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Name this state...'), { target: { value: 'my-snap' } })
    fireEvent.click(screen.getByText('Save'))
    expect(soulsController.saveWeightSnapshot).toHaveBeenCalledWith('my-snap')
  })

  it('loads snapshot on Load click', async () => {
    const onTraitsChanged = vi.fn().mockResolvedValue(undefined)
    ;(soulsController.listWeightSnapshots as any).mockResolvedValue([{ name: 'snap1', saved_at: '2025-01-01' }])
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={onTraitsChanged} />)
    fireEvent.change(screen.getByPlaceholderText('Name this state...'), { target: { value: 'trigger-save' } })
    fireEvent.click(screen.getByText('Save'))
    const loadBtn = await screen.findByText('Load')
    fireEvent.click(loadBtn)
    expect(soulsController.loadWeightSnapshot).toHaveBeenCalledWith('snap1')
  })

  it('shows toast on save error', async () => {
    ;(soulsController.saveWeightSnapshot as any).mockRejectedValue(new Error('disk full'))
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Name this state...'), { target: { value: 'fail' } })
    fireEvent.click(screen.getByText('Save'))
    await vi.waitFor(() => expect(mockAddToast).toHaveBeenCalledWith('disk full', 'error'))
  })

  it('shows toast on load error', async () => {
    ;(soulsController.loadWeightSnapshot as any).mockRejectedValue(new Error('not found'))
    ;(soulsController.listWeightSnapshots as any).mockResolvedValue([{ name: 'broken' }])
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Name this state...'), { target: { value: 'trigger' } })
    fireEvent.click(screen.getByText('Save'))
    const loadBtn = await screen.findByText('Load')
    fireEvent.click(loadBtn)
    await vi.waitFor(() => expect(mockAddToast).toHaveBeenCalledWith('not found', 'error'))
  })

  it('shows formatted date for snapshots', async () => {
    ;(soulsController.listWeightSnapshots as any).mockResolvedValue([{ name: 's1', saved_at: '2025-06-15T10:30:00Z' }])
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Name this state...'), { target: { value: 'trigger' } })
    fireEvent.click(screen.getByText('Save'))
    expect(await screen.findByText('s1')).toBeDefined()
    expect(screen.getByText(/Jun 15/)).toBeDefined()
  })

  it('passes currentSoulName to SoulVisualizer', () => {
    render(<PersonalityProfileCard traitWeights={defaultWeights} currentSoulName="wise" onTraitsSaved={vi.fn()} onTraitsChanged={vi.fn()} />)
    expect(screen.getByTestId('soul-visualizer').getAttribute('data-name')).toBe('wise')
  })

  it('calls onTraitsSaved from TraitEditor', () => {
    const onSaved = vi.fn().mockResolvedValue(undefined)
    render(<PersonalityProfileCard traitWeights={defaultWeights} onTraitsSaved={onSaved} onTraitsChanged={vi.fn()} />)
    fireEvent.click(screen.getByText('Edit'))
    fireEvent.click(screen.getByText('save'))
    expect(onSaved).toHaveBeenCalledWith({ personality: { warm: 0.9 } })
  })
})
