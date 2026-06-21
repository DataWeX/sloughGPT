/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SearchBox, SectionTabs, ActionButton, EmptyState } from './search'

afterEach(cleanup)

describe('SearchBox', () => {
  it('renders input with placeholder', () => {
    render(<SearchBox value="" onChange={() => {}} />)
    expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument()
  })

  it('renders custom placeholder', () => {
    render(<SearchBox value="" onChange={() => {}} placeholder="Find..." />)
    expect(screen.getByPlaceholderText('Find...')).toBeInTheDocument()
  })

  it('displays current value', () => {
    render(<SearchBox value="hello" onChange={() => {}} />)
    const input = screen.getByRole('textbox') as HTMLInputElement
    expect(input.value).toBe('hello')
  })

  it('calls onChange when typing', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<SearchBox value="" onChange={onChange} />)
    await user.type(screen.getByRole('textbox'), 'a')
    expect(onChange).toHaveBeenCalledWith('a')
  })

  it('shows clear button when value is non-empty', () => {
    render(<SearchBox value="text" onChange={() => {}} />)
    expect(screen.getByLabelText('Clear search')).toBeInTheDocument()
  })

  it('hides clear button when value is empty', () => {
    render(<SearchBox value="" onChange={() => {}} />)
    expect(screen.queryByLabelText('Clear search')).not.toBeInTheDocument()
  })

  it('calls onChange with empty on clear', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<SearchBox value="text" onChange={onChange} />)
    await user.click(screen.getByLabelText('Clear search'))
    expect(onChange).toHaveBeenCalledWith('')
  })

  it('renders search icon', () => {
    const { container } = render(<SearchBox value="" onChange={() => {}} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })
})

describe('SectionTabs', () => {
  const options = [
    { value: 'all', label: 'All' },
    { value: 'active', label: 'Active', count: 5 },
    { value: 'archived', label: 'Archived' },
  ]

  it('renders all tab labels', () => {
    render(<SectionTabs value="all" onChange={() => {}} options={options} />)
    expect(screen.getByText('All')).toBeInTheDocument()
    expect(screen.getByText('Active')).toBeInTheDocument()
    expect(screen.getByText('Archived')).toBeInTheDocument()
  })

  it('highlights the active tab', () => {
    render(<SectionTabs value="active" onChange={() => {}} options={options} />)
    const activeBtn = screen.getByText('Active').closest('button')
    expect(activeBtn).toHaveClass('bg-background')
    const inactiveBtn = screen.getByText('All').closest('button')
    expect(inactiveBtn).toHaveClass('text-muted-foreground')
  })

  it('shows count when provided and > 0', () => {
    render(<SectionTabs value="all" onChange={() => {}} options={options} />)
    expect(screen.getByText('5')).toBeInTheDocument()
  })

  it('does not show count when 0', () => {
    const zeroOpts = [{ value: 'a', label: 'A', count: 0 }]
    render(<SectionTabs value="a" onChange={() => {}} options={zeroOpts} />)
    expect(screen.queryByText('0')).not.toBeInTheDocument()
  })

  it('calls onChange on click', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<SectionTabs value="all" onChange={onChange} options={options} />)
    await user.click(screen.getByText('Active'))
    expect(onChange).toHaveBeenCalledWith('active')
  })

  it('renders icon when provided', () => {
    const opts = [{ value: 'x', label: 'With Icon', icon: <span data-testid="tab-icon" /> }]
    render(<SectionTabs value="x" onChange={() => {}} options={opts} />)
    expect(screen.getByTestId('tab-icon')).toBeInTheDocument()
  })
})

describe('ActionButton', () => {
  it('renders label', () => {
    render(<ActionButton icon={<span />} label="Save" onClick={() => {}} />)
    expect(screen.getByText('Save')).toBeInTheDocument()
  })

  it('renders icon', () => {
    render(<ActionButton icon={<span data-testid="action-icon" />} label="Go" onClick={() => {}} />)
    expect(screen.getByTestId('action-icon')).toBeInTheDocument()
  })

  it('calls onClick when clicked', async () => {
    const onClick = vi.fn()
    const user = userEvent.setup()
    render(<ActionButton icon={<span />} label="Click" onClick={onClick} />)
    await user.click(screen.getByText('Click'))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('applies default variant as button', () => {
    const { container } = render(<ActionButton icon={<span />} label="Default" onClick={() => {}} />)
    expect(container.firstChild).toHaveClass('bg-primary')
  })

  it('applies ghost variant', () => {
    const { container } = render(<ActionButton icon={<span />} label="Ghost" onClick={() => {}} variant="ghost" />)
    expect(container.firstChild).toHaveClass('text-muted-foreground')
  })

  it('applies destructive variant', () => {
    const { container } = render(<ActionButton icon={<span />} label="Delete" onClick={() => {}} variant="destructive" />)
    expect(container.firstChild).toHaveClass('text-destructive')
  })
})

describe('EmptyState (search)', () => {
  it('renders default message', () => {
    render(<EmptyState />)
    expect(screen.getByText('No items')).toBeInTheDocument()
  })

  it('renders custom message', () => {
    render(<EmptyState message="Nothing found" />)
    expect(screen.getByText('Nothing found')).toBeInTheDocument()
  })

  it('renders action button when provided', () => {
    render(<EmptyState action={{ label: 'Retry', onClick: () => {} }} />)
    expect(screen.getByText('Retry')).toBeInTheDocument()
  })

  it('calls action onClick', async () => {
    const onClick = vi.fn()
    const user = userEvent.setup()
    render(<EmptyState action={{ label: 'Try again', onClick }} />)
    await user.click(screen.getByText('Try again'))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('does not render action when not provided', () => {
    render(<EmptyState />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})
