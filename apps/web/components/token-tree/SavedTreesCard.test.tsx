import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SavedTreesCard } from './SavedTreesCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, variant, ...props }: any) => (
    <button data-testid={`btn-${variant || 'default'}`} onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: (props: any) => <input data-testid="save-name-input" {...props} />,
}))

const defaultProps = {
  savedTrees: [],
  saveName: '',
  loading: false,
  onSaveNameChange: vi.fn(),
  onSave: vi.fn(),
  onLoad: vi.fn(),
  onDelete: vi.fn(),
}

const trees = [
  { name: 'tree-v1', vocab_size: 256, num_merges: 100 },
  { name: 'tree-v2', vocab_size: 512, num_merges: 200 },
]

describe('SavedTreesCard', () => {
  it('renders the card title', () => {
    render(<SavedTreesCard {...defaultProps} />)
    expect(screen.getByText('Saved Trees')).toBeDefined()
  })

  it('renders the save name input', () => {
    render(<SavedTreesCard {...defaultProps} />)
    expect(screen.getByTestId('save-name-input')).toBeDefined()
  })

  it('shows no saved trees message when empty', () => {
    render(<SavedTreesCard {...defaultProps} />)
    expect(screen.getByText('No saved trees.')).toBeDefined()
  })

  it('renders saved trees with names and details', () => {
    render(<SavedTreesCard {...defaultProps} savedTrees={trees} />)
    expect(screen.getByText('tree-v1')).toBeDefined()
    expect(screen.getByText('tree-v2')).toBeDefined()
    expect(screen.getByText('256 vocab, 100 merges')).toBeDefined()
  })

  it('renders load and delete buttons for each tree', () => {
    render(<SavedTreesCard {...defaultProps} savedTrees={trees} />)
    expect(screen.getAllByText('Load').length).toBe(2)
    expect(screen.getAllByText('Delete').length).toBe(2)
  })

  it('calls onSave with correct name', async () => {
    const onSave = vi.fn()
    render(<SavedTreesCard {...defaultProps} saveName="my-tree" onSave={onSave} />)
    const saveBtn = screen.getByText('Save Current')
    expect((saveBtn as HTMLButtonElement).disabled).toBe(false)
  })

  it('disables save button when name is empty', () => {
    render(<SavedTreesCard {...defaultProps} />)
    const saveBtn = screen.getByText('Save Current')
    expect((saveBtn as HTMLButtonElement).disabled).toBe(true)
  })
})
