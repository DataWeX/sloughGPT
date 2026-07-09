/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Avatar, AvatarGroup, ProgressBar, Spinner, Divider, CardDeck, EmptyState, SearchField, Pagination } from '@sloughgpt/strui'

afterEach(cleanup)

describe('Avatar', () => {
  it('renders fallback text when no src', () => {
    render(<Avatar fallback="JD" />)
    expect(screen.getByText('JD')).toBeInTheDocument()
  })

  it('renders img when src provided', () => {
    render(<Avatar src="/avatar.png" alt="User" fallback="U" />)
    const img = screen.getByRole('img')
    expect(img).toHaveAttribute('src', '/avatar.png')
    expect(img).toHaveAttribute('alt', 'User')
  })

  it('does not render fallback when src provided', () => {
    render(<Avatar src="/img.png" alt="A" fallback="X" />)
    expect(screen.queryByText('X')).not.toBeInTheDocument()
  })

  it('applies size classes', () => {
    const { rerender, container } = render(<Avatar fallback="A" size="sm" />)
    expect(container.firstChild).toHaveClass('h-6')
    rerender(<Avatar fallback="A" size="lg" />)
    expect(container.firstChild).toHaveClass('h-10')
  })

  it('applies custom className', () => {
    const { container } = render(<Avatar fallback="A" className="ring-2" />)
    expect(container.firstChild).toHaveClass('ring-2')
  })

  it('is rounded-full', () => {
    const { container } = render(<Avatar fallback="A" />)
    expect(container.firstChild).toHaveClass('rounded-full')
  })
})

describe('AvatarGroup', () => {
  const users = [
    { fallback: 'A' },
    { fallback: 'B' },
    { fallback: 'C' },
  ]

  it('renders all avatars up to max', () => {
    render(<AvatarGroup avatars={users} max={3} />)
    expect(screen.getByText('A')).toBeInTheDocument()
    expect(screen.getByText('B')).toBeInTheDocument()
    expect(screen.getByText('C')).toBeInTheDocument()
  })

  it('shows overflow count when exceeding max', () => {
    render(<AvatarGroup avatars={[...users, { fallback: 'D' }, { fallback: 'E' }]} max={3} />)
    expect(screen.getByText('+2')).toBeInTheDocument()
  })

  it('hides overflow when within max', () => {
    render(<AvatarGroup avatars={users} max={3} />)
    expect(screen.queryByText('+')).not.toBeInTheDocument()
  })
})

describe('ProgressBar', () => {
  it('renders percentage text', () => {
    render(<ProgressBar value={75} />)
    expect(screen.getByText('75%')).toBeInTheDocument()
  })

  it('renders label when provided', () => {
    render(<ProgressBar value={50} label="Training" />)
    expect(screen.getByText('Training')).toBeInTheDocument()
  })

  it('hides value when showValue is false', () => {
    render(<ProgressBar value={50} showValue={false} />)
    expect(screen.queryByText('50%')).not.toBeInTheDocument()
  })

  it('clamps to 100%', () => {
    render(<ProgressBar value={200} />)
    expect(screen.getByText('100%')).toBeInTheDocument()
  })

  it('applies success variant', () => {
    const { container } = render(<ProgressBar value={50} variant="success" />)
    const fill = container.querySelector('.bg-success')
    expect(fill).toBeInTheDocument()
  })

  it('applies error variant', () => {
    const { container } = render(<ProgressBar value={50} variant="error" />)
    const fill = container.querySelector('.bg-destructive')
    expect(fill).toBeInTheDocument()
  })
})

describe('Spinner', () => {
  it('renders an SVG element', () => {
    const { container } = render(<Spinner />)
    const svg = container.querySelector('svg')
    expect(svg).toBeInTheDocument()
  })

  it('has animate-spin class', () => {
    const { container } = render(<Spinner />)
    expect(container.firstChild).toHaveClass('animate-spin')
  })

  it('applies size classes', () => {
    const { rerender, container } = render(<Spinner size="sm" />)
    expect(container.firstChild).toHaveClass('w-4')
    rerender(<Spinner size="lg" />)
    expect(container.firstChild).toHaveClass('w-8')
  })
})

describe('Divider', () => {
  it('renders a horizontal line', () => {
    const { container } = render(<Divider />)
    expect(container.firstChild).toHaveClass('h-px')
    expect(container.firstChild).toHaveClass('bg-border')
  })

  it('renders label between lines', () => {
    render(<Divider label="OR" />)
    expect(screen.getByText('OR')).toBeInTheDocument()
  })

  it('has flex layout when label is provided', () => {
    const { container } = render(<Divider label="Section" />)
    expect(container.firstChild).toHaveClass('flex')
    expect(container.firstChild).toHaveClass('items-center')
  })
})

describe('CardDeck', () => {
  it('renders children', () => {
    render(<CardDeck><span data-testid="deck-child" /></CardDeck>)
    expect(screen.getByTestId('deck-child')).toBeInTheDocument()
  })

  it('renders title when provided', () => {
    render(<CardDeck title="Settings">content</CardDeck>)
    expect(screen.getByText('Settings')).toBeInTheDocument()
  })

  it('renders description when provided', () => {
    render(<CardDeck description="Configure options">content</CardDeck>)
    expect(screen.getByText('Configure options')).toBeInTheDocument()
  })

  it('renders footer when provided', () => {
    render(<CardDeck footer={<button>Save</button>}>content</CardDeck>)
    expect(screen.getByText('Save')).toBeInTheDocument()
  })

  it('does not render header when no title/description', () => {
    const { container } = render(<CardDeck>content</CardDeck>)
    const headers = container.querySelectorAll('.border-b')
    expect(headers.length).toBe(0)
  })
})

describe('EmptyState (specialized)', () => {
  it('renders title', () => {
    render(<EmptyState title="Nothing here" />)
    expect(screen.getByText('Nothing here')).toBeInTheDocument()
  })

  it('renders description when provided', () => {
    render(<EmptyState title="Empty" description="No items found" />)
    expect(screen.getByText('No items found')).toBeInTheDocument()
  })

  it('renders icon when provided', () => {
    render(<EmptyState title="Title" icon={<span data-testid="empty-icon" />} />)
    expect(screen.getByTestId('empty-icon')).toBeInTheDocument()
  })

  it('renders action when provided', () => {
    render(<EmptyState title="Title" action={<button>Create</button>} />)
    expect(screen.getByText('Create')).toBeInTheDocument()
  })
})

describe('SearchField', () => {
  it('renders input with placeholder', () => {
    render(<SearchField value="" onChange={() => {}} />)
    expect(screen.getByPlaceholderText('Search…')).toBeInTheDocument()
  })

  it('displays current value', () => {
    render(<SearchField value="query" onChange={() => {}} />)
    const input = screen.getByRole('searchbox') as HTMLInputElement
    expect(input.value).toBe('query')
  })

  it('calls onChange when typing', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<SearchField value="" onChange={onChange} />)
    await user.type(screen.getByRole('searchbox'), 'x')
    expect(onChange).toHaveBeenCalledWith('x')
  })

  it('renders search icon', () => {
    const { container } = render(<SearchField value="" onChange={() => {}} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })
})

describe('Pagination', () => {
  it('renders page info', () => {
    render(<Pagination page={2} total={50} pageSize={10} onChange={() => {}} />)
    expect(screen.getByText(/Page 2 of 5/)).toBeInTheDocument()
  })

  it('renders Prev and Next buttons', () => {
    render(<Pagination page={2} total={50} pageSize={10} onChange={() => {}} />)
    expect(screen.getByText('Prev')).toBeInTheDocument()
    expect(screen.getByText('Next')).toBeInTheDocument()
  })

  it('disables Prev on first page', () => {
    render(<Pagination page={1} total={50} pageSize={10} onChange={() => {}} />)
    expect(screen.getByText('Prev')).toBeDisabled()
  })

  it('disables Next on last page', () => {
    render(<Pagination page={5} total={50} pageSize={10} onChange={() => {}} />)
    expect(screen.getByText('Next')).toBeDisabled()
  })

  it('returns null when totalPages is 1', () => {
    const { container } = render(<Pagination page={1} total={5} pageSize={10} onChange={() => {}} />)
    expect(container.firstChild).toBeNull()
  })

  it('calls onChange with prev page', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Pagination page={3} total={50} pageSize={10} onChange={onChange} />)
    await user.click(screen.getByText('Prev'))
    expect(onChange).toHaveBeenCalledWith(2)
  })

  it('calls onChange with next page', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Pagination page={3} total={50} pageSize={10} onChange={onChange} />)
    await user.click(screen.getByText('Next'))
    expect(onChange).toHaveBeenCalledWith(4)
  })
})
