import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { ThemePreview } from './ThemePreview'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => (
    <button data-testid="button" onClick={onClick} {...props}>{children}</button>
  ),
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
}))

afterEach(() => cleanup())

describe('ThemePreview', () => {
  const themes = [
    { name: 'blue', label: 'Blue', rgb: '90 130 220' },
    { name: 'purple', label: 'Purple', rgb: '155 108 214' },
  ]

  it('renders the card title', () => {
    render(<ThemePreview themes={themes} />)
    expect(screen.getByText('Theme in Context')).toBeDefined()
  })

  it('renders all theme labels', () => {
    render(<ThemePreview themes={themes} />)
    expect(screen.getByText('Blue')).toBeDefined()
    expect(screen.getByText('Purple')).toBeDefined()
  })

  it('renders theme class names', () => {
    render(<ThemePreview themes={themes} />)
    expect(screen.getByText('html.theme-blue')).toBeDefined()
    expect(screen.getByText('html.theme-purple')).toBeDefined()
  })

  it('renders Apply buttons for each theme', () => {
    render(<ThemePreview themes={themes} />)
    const applyButtons = screen.getAllByText('Apply')
    expect(applyButtons.length).toBe(2)
  })

  it('calls onApply with theme name when Apply is clicked', () => {
    const onApply = vi.fn()
    render(<ThemePreview themes={themes} onApply={onApply} />)
    fireEvent.click(screen.getAllByText('Apply')[0])
    expect(onApply).toHaveBeenCalledWith('blue')
  })
})
