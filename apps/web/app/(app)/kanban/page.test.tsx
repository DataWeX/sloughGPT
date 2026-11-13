import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const fetchBoard = vi.hoisted(() => vi.fn())
const transitionCard = vi.hoisted(() => vi.fn())

vi.mock('@/lib/planner-client', () => ({
  fetchBoard,
  transitionCard,
}))

const fixtureBoard = {
  name: 'Main',
  columns: [
    { name: 'todo', label: 'Todo', wip_limit: 5, order: 0 },
    { name: 'wip', label: 'WIP', wip_limit: 3, order: 1 },
    { name: 'review', label: 'Review', wip_limit: 2, order: 2 },
    { name: 'done', label: 'Done', wip_limit: 0, order: 3 },
  ],
  cards: [
    {
      id: 'card-1',
      title: 'First task',
      description: '',
      column: 'todo',
      priority: 'medium',
      tags: ['frontend'],
      due_date: '',
      assignee: '',
      sprint: '',
      gh: '',
      notes: [],
      created_at: '',
      updated_at: '',
      root_hash: '',
    },
    {
      id: 'card-2',
      title: 'Second task',
      description: '',
      column: 'wip',
      priority: 'high',
      tags: [],
      due_date: '',
      assignee: '',
      sprint: '',
      gh: '',
      notes: [],
      created_at: '',
      updated_at: '',
      root_hash: '',
    },
  ],
}

async function renderPage() {
  const mod = await import('./page')
  render(<mod.default />)
  await waitFor(() => expect(fetchBoard).toHaveBeenCalled())
  await screen.findByText('First task')
}

beforeEach(() => {
  vi.clearAllMocks()
  fetchBoard.mockResolvedValue({ board: fixtureBoard })
})

describe('Kanban page', () => {
  it('renders columns and cards from the board API (DOM journey)', async () => {
    await renderPage()
    expect(screen.getByText('First task')).toBeTruthy()
    expect(screen.getByText('Second task')).toBeTruthy()
    expect(screen.getByText('Todo')).toBeTruthy()
  })

  it('selecting a card shows engine-derived adjacent targets only', async () => {
    await renderPage()
    fireEvent.click(screen.getByText('First task'))
    // todo's only lane neighbour is wip — review/done must NOT be offered
    const wipButton = await screen.findByText('→ wip')
    expect(wipButton).toBeTruthy()
    expect(screen.queryByText('→ review')).toBeNull()
    expect(screen.queryByText('→ done')).toBeNull()
  })

  it('done cards expose no outbound moves (terminal)', async () => {
    fetchBoard.mockResolvedValue({
      board: {
        ...fixtureBoard,
        cards: [{ ...fixtureBoard.cards[1], id: 'card-d', column: 'done' }],
      },
    })
    const mod = await import('./page')
    render(<mod.default />)
    await waitFor(() => expect(fetchBoard).toHaveBeenCalled())
    const doneCard = await screen.findByText('Second task')
    fireEvent.click(doneCard)
    expect(screen.queryByText('→ wip')).toBeNull()
    expect(screen.queryByText('→ review')).toBeNull()
    expect(screen.getByText('Cancel')).toBeTruthy()
  })

  it('falls back to DOM view with feedback when WebGL is unavailable', async () => {
    await renderPage()
    fireEvent.click(screen.getByRole('button', { name: '3D' }))
    // jsdom has no WebGL — the toggle must refuse, explain, and keep the list
    await screen.findByText(/WebGL is not available/)
    expect(screen.getByText('First task')).toBeTruthy()
    expect(screen.getByRole('button', { name: '3D' })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('button', { name: 'List' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('passes the live board into the 3D scene when WebGL exists', async () => {
    const original = HTMLCanvasElement.prototype.getContext
    HTMLCanvasElement.prototype.getContext = vi.fn(
      () => ({}),
    ) as unknown as typeof HTMLCanvasElement.prototype.getContext

    vi.doMock('@/components/kanban3d/KanbanScene', () => ({
      default: ({ board }: { board: typeof fixtureBoard }) => (
        <div data-testid="scene3d">{`${board.cards.length} cards in scene`}</div>
      ),
    }))
    vi.resetModules()

    try {
      const mod = await import('./page')
      render(<mod.default />)
      await waitFor(() => expect(fetchBoard).toHaveBeenCalled())
      await screen.findByText('First task')
      fireEvent.click(screen.getByRole('button', { name: '3D' }))
      await screen.findByTestId('scene3d')
      expect(screen.getByText('2 cards in scene')).toBeTruthy()
    } finally {
      HTMLCanvasElement.prototype.getContext = original
      vi.doUnmock('@/components/kanban3d/KanbanScene')
      vi.resetModules()
    }
  })

  it('3D scene selection drives the shared action bar with engine targets', async () => {
    const original = HTMLCanvasElement.prototype.getContext
    HTMLCanvasElement.prototype.getContext = vi.fn(
      () => ({}),
    ) as unknown as typeof HTMLCanvasElement.prototype.getContext

    vi.doMock('@/components/kanban3d/KanbanScene', () => ({
      default: ({ onSelect }: { onSelect: (id: string) => void }) => (
        <button data-testid="scene3d" onClick={() => onSelect('card-1')}>
          pick first
        </button>
      ),
    }))
    vi.resetModules()

    try {
      const mod = await import('./page')
      render(<mod.default />)
      await waitFor(() => expect(fetchBoard).toHaveBeenCalled())
      fireEvent.click(screen.getByRole('button', { name: '3D' }))
      fireEvent.click(await screen.findByTestId('scene3d'))
      // selecting in 3D must surface the same adjacent-targets bar as the DOM view
      await screen.findByText('→ wip')
      expect(screen.queryByText('→ review')).toBeNull()
      expect(screen.queryByText('→ done')).toBeNull()
      expect(screen.getByText('Cancel')).toBeTruthy()
    } finally {
      HTMLCanvasElement.prototype.getContext = original
      vi.doUnmock('@/components/kanban3d/KanbanScene')
      vi.resetModules()
    }
  })
})
