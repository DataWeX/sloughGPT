/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Chip, Chips, Badge, TagInput } from './tags'

afterEach(cleanup)

describe('Chip', () => {
  it('renders label text', () => {
    render(<Chip label="React" />)
    expect(screen.getByText('React')).toBeInTheDocument()
  })

  it('renders as button when onClick provided', () => {
    render(<Chip label="Clickable" onClick={() => {}} />)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('renders as span when no onClick', () => {
    const { container } = render(<Chip label="Static" />)
    expect(container.querySelector('button')).not.toBeInTheDocument()
    expect(container.querySelector('span')?.tagName).toBe('SPAN')
  })

  it('applies selected styles', () => {
    const { container } = render(<Chip label="Selected" selected />)
    const el = container.querySelector('span')!
    expect(el.className).toContain('bg-primary')
    expect(el.className).toContain('text-primary-foreground')
  })

  it('applies outline variant', () => {
    const { container } = render(<Chip label="Outline" variant="outline" />)
    const el = container.querySelector('span')!
    expect(el.className).toContain('border')
    expect(el.className).toContain('border-border')
  })

  it('applies sm size', () => {
    const { container } = render(<Chip label="Small" size="sm" />)
    const el = container.querySelector('span')!
    expect(el.className).toContain('text-[10px]')
  })

  it('renders icon when provided', () => {
    render(<Chip label="With Icon" icon={<span data-testid="chip-icon" />} />)
    expect(screen.getByTestId('chip-icon')).toBeInTheDocument()
  })

  it('shows remove button when removable', () => {
    const { container } = render(<Chip label="Removable" removable />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  it('calls onClick when clicked', async () => {
    const onClick = vi.fn()
    const user = userEvent.setup()
    render(<Chip label="Tap" onClick={onClick} />)
    await user.click(screen.getByText('Tap'))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('calls onRemove when remove button clicked', async () => {
    const onRemove = vi.fn()
    const user = userEvent.setup()
    render(<Chip label="X" removable onRemove={onRemove} />)
    await user.click(screen.getByRole('button'))
    expect(onRemove).toHaveBeenCalledOnce()
  })
})

describe('Chips', () => {
  const options = [
    { value: 'a', label: 'Option A' },
    { value: 'b', label: 'Option B' },
    { value: 'c', label: 'Option C' },
  ]

  it('renders all options', () => {
    render(<Chips value={[]} onChange={() => {}} options={options} />)
    expect(screen.getByText('Option A')).toBeInTheDocument()
    expect(screen.getByText('Option B')).toBeInTheDocument()
    expect(screen.getByText('Option C')).toBeInTheDocument()
  })

  it('marks selected options', () => {
    render(<Chips value={['a', 'c']} onChange={() => {}} options={options} />)
    expect(screen.getByText('Option A').closest('button')).toHaveClass('bg-primary')
    expect(screen.getByText('Option B').closest('button')).not.toHaveClass('bg-primary')
    expect(screen.getByText('Option C').closest('button')).toHaveClass('bg-primary')
  })

  it('adds option on click', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Chips value={['a']} onChange={onChange} options={options} />)
    await user.click(screen.getByText('Option B'))
    expect(onChange).toHaveBeenCalledWith(['a', 'b'])
  })

  it('removes option on click if already selected', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Chips value={['a', 'b']} onChange={onChange} options={options} />)
    await user.click(screen.getByText('Option A'))
    expect(onChange).toHaveBeenCalledWith(['b'])
  })

  it('respects max limit', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Chips value={['a', 'b']} onChange={onChange} options={options} max={2} />)
    await user.click(screen.getByText('Option C'))
    expect(onChange).not.toHaveBeenCalled()
  })
})

describe('Badge (tags)', () => {
  it('renders label', () => {
    render(<Badge label="Active" />)
    expect(screen.getByText('Active')).toBeInTheDocument()
  })

  it('applies default variant', () => {
    const { container } = render(<Badge label="Default" />)
    expect(container.firstChild).toHaveClass('bg-primary/10')
  })

  it('applies success variant', () => {
    const { container } = render(<Badge label="Done" variant="success" />)
    expect(container.firstChild).toHaveClass('bg-success/10')
  })

  it('applies warning variant', () => {
    const { container } = render(<Badge label="Caution" variant="warning" />)
    expect(container.firstChild).toHaveClass('bg-warning/10')
  })

  it('applies error variant', () => {
    const { container } = render(<Badge label="Fail" variant="error" />)
    expect(container.firstChild).toHaveClass('bg-destructive/10')
  })

  it('applies outline variant', () => {
    const { container } = render(<Badge label="Outline" variant="outline" />)
    expect(container.firstChild).toHaveClass('border')
  })

  it('applies sm size', () => {
    const { container } = render(<Badge label="Small" size="sm" />)
    expect(container.firstChild).toHaveClass('text-[10px]')
  })

  it('applies custom className', () => {
    const { container } = render(<Badge label="Custom" className="mx-2" />)
    expect(container.firstChild).toHaveClass('mx-2')
  })
})

describe('TagInput', () => {
  it('renders input with placeholder', () => {
    render(<TagInput value={[]} onChange={() => {}} placeholder="Add tag..." />)
    expect(screen.getByPlaceholderText('Add tag...')).toBeInTheDocument()
  })

  it('shows existing tags', () => {
    render(<TagInput value={['react', 'typescript']} onChange={() => {}} />)
    expect(screen.getByText('react')).toBeInTheDocument()
    expect(screen.getByText('typescript')).toBeInTheDocument()
  })

  it('adds tag on Enter', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<TagInput value={[]} onChange={onChange} />)
    const input = screen.getByPlaceholderText('Add tag...')
    await user.type(input, 'newtag{Enter}')
    expect(onChange).toHaveBeenCalledWith(['newtag'])
  })

  it('does not add empty tag', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<TagInput value={[]} onChange={onChange} />)
    const input = screen.getByPlaceholderText('Add tag...')
    await user.type(input, '   {Enter}')
    expect(onChange).not.toHaveBeenCalled()
  })

  it('does not add duplicate tag', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<TagInput value={['existing']} onChange={onChange} />)
    const input = screen.getByDisplayValue('')
    await user.type(input, 'existing{Enter}')
    expect(onChange).not.toHaveBeenCalled()
  })

  it('clears input after adding', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<TagInput value={[]} onChange={onChange} />)
    const input = screen.getByPlaceholderText('Add tag...') as HTMLInputElement
    await user.type(input, 'tag{Enter}')
    expect(input.value).toBe('')
  })

  it('removes tag via Chip onRemove', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<TagInput value={['remove-me']} onChange={onChange} />)
    const removeBtns = screen.getAllByRole('button')
    await user.click(removeBtns[0])
    expect(onChange).toHaveBeenCalledWith([])
  })
})
