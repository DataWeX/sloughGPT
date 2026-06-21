/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import * as Icons from './icons'
import { Tooltip, Dropdown, Menu, MenuItem, MenuDivider } from './menu'

afterEach(cleanup)

const iconEntries = Object.entries(Icons).filter(([name]) => name.startsWith('Icon'))

describe('Icons', () => {
  it.each(iconEntries)('%s renders an SVG element', (_name, Component) => {
    const { container } = render(<Component />)
    const svg = container.querySelector('svg')
    expect(svg).toBeInTheDocument()
    expect(svg).toHaveAttribute('viewBox', '0 0 24 24')
  })

  it.each(iconEntries)('%s applies className', (_name, Component) => {
    const { container } = render(<Component className="custom-icon" />)
    const svg = container.querySelector('svg')
    expect(svg).toHaveClass('custom-icon')
  })

  it('IconStar renders filled variant', () => {
    const { container } = render(<Icons.IconStar filled />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  it('IconHeart renders filled variant', () => {
    const { container } = render(<Icons.IconHeart filled />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })
})

describe('Tooltip', () => {
  it('renders trigger children', () => {
    render(<Tooltip content="Help text"><button>Hover me</button></Tooltip>)
    expect(screen.getByText('Hover me')).toBeInTheDocument()
  })

  it('renders tooltip content', () => {
    render(<Tooltip content="Help text"><span>Target</span></Tooltip>)
    expect(screen.getByText('Help text')).toBeInTheDocument()
  })

  it('has group class for hover behavior', () => {
    const { container } = render(<Tooltip content="tip"><span>X</span></Tooltip>)
    expect(container.firstChild).toHaveClass('group')
  })
})

describe('MenuItem', () => {
  it('renders children', () => {
    render(<MenuItem>Save</MenuItem>)
    expect(screen.getByText('Save')).toBeInTheDocument()
  })

  it('renders as button', () => {
    render(<MenuItem>Click</MenuItem>)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('calls onClick when clicked', async () => {
    const onClick = vi.fn()
    const user = userEvent.setup()
    render(<MenuItem onClick={onClick}>Action</MenuItem>)
    await user.click(screen.getByText('Action'))
    expect(onClick).toHaveBeenCalledOnce()
  })
})

describe('MenuDivider', () => {
  it('renders a horizontal line', () => {
    const { container } = render(<MenuDivider />)
    expect(container.firstChild).toHaveClass('h-px')
    expect(container.firstChild).toHaveClass('bg-border/50')
  })
})

describe('Dropdown', () => {
  it('renders trigger', () => {
    render(<Dropdown trigger={<button>Open</button>}>content</Dropdown>)
    expect(screen.getByText('Open')).toBeInTheDocument()
  })

  it('renders children', () => {
    render(<Dropdown trigger={<span>Menu</span>}><span>Item</span></Dropdown>)
    expect(screen.getByText('Item')).toBeInTheDocument()
  })

  it('has group class for hover', () => {
    const { container } = render(<Dropdown trigger={<span>T</span>}>c</Dropdown>)
    expect(container.firstChild).toHaveClass('group')
  })
})

describe('Menu', () => {
  it('renders trigger', () => {
    render(<Menu trigger={<button>Options</button>}><MenuItem>Edit</MenuItem></Menu>)
    expect(screen.getByText('Options')).toBeInTheDocument()
  })

  it('renders menu items', () => {
    render(<Menu trigger={<span>M</span>}><MenuItem>Edit</MenuItem><MenuItem>Delete</MenuItem></Menu>)
    expect(screen.getByText('Edit')).toBeInTheDocument()
    expect(screen.getByText('Delete')).toBeInTheDocument()
  })

  it('has group class for hover behavior', () => {
    const { container } = render(<Menu trigger={<span>T</span>}><span>X</span></Menu>)
    expect(container.firstChild).toHaveClass('group')
  })

  it('renders with MenuDivider separator', () => {
    render(<Menu trigger={<span>M</span>}><MenuItem>Copy</MenuItem><MenuDivider /><MenuItem>Paste</MenuItem></Menu>)
    expect(screen.getByText('Copy')).toBeInTheDocument()
    expect(screen.getByText('Paste')).toBeInTheDocument()
  })
})
