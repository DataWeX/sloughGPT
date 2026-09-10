import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { ColorSwatchGrid } from './ColorSwatchGrid'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
}))

afterEach(() => cleanup())

describe('ColorSwatchGrid', () => {
  const colors = [
    { name: '--primary', rgb: '124 82 196', label: 'Primary', desc: 'Main color' },
    { name: '--accent', rgb: '236 145 95', label: 'Accent', desc: 'Highlight' },
  ]

  it('renders the grid title', () => {
    render(<ColorSwatchGrid title="Light Mode" colors={colors} />)
    expect(screen.getByText('Light Mode')).toBeDefined()
  })

  it('renders all color labels', () => {
    render(<ColorSwatchGrid title="Palette" colors={colors} />)
    expect(screen.getByText('Primary')).toBeDefined()
    expect(screen.getByText('Accent')).toBeDefined()
  })

  it('renders RGB values', () => {
    render(<ColorSwatchGrid title="Palette" colors={colors} />)
    expect(screen.getByText('124 82 196')).toBeDefined()
    expect(screen.getByText('236 145 95')).toBeDefined()
  })

  it('renders descriptions', () => {
    render(<ColorSwatchGrid title="Palette" colors={colors} />)
    expect(screen.getByText('Main color')).toBeDefined()
    expect(screen.getByText('Highlight')).toBeDefined()
  })

  it('renders a card wrapper', () => {
    render(<ColorSwatchGrid title="Test" colors={colors} />)
    expect(screen.getAllByTestId('card').length).toBeGreaterThanOrEqual(1)
  })
})
