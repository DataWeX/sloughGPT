/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { ShortcutLegendCard, type LegendItem } from './ShortcutLegendCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const items: LegendItem[] = [
  { key: 'Ctrl+K', description: 'Open command palette' },
  { key: 'R', description: 'Refresh data' },
]

describe('ShortcutLegendCard', () => {
  it('renders the title', () => {
    render(<ShortcutLegendCard items={[]} />)
    expect(screen.getByText('Keyboard Shortcuts')).toBeDefined()
  })

  it('renders custom title', () => {
    render(<ShortcutLegendCard title="Custom Title" items={[]} />)
    expect(screen.getByText('Custom Title')).toBeDefined()
  })

  it('renders legend key bindings', () => {
    render(<ShortcutLegendCard items={items} />)
    expect(screen.getByText('Ctrl+K')).toBeDefined()
    expect(screen.getByText('R')).toBeDefined()
  })

  it('renders legend descriptions', () => {
    render(<ShortcutLegendCard items={items} />)
    expect(screen.getByText('Open command palette')).toBeDefined()
    expect(screen.getByText('Refresh data')).toBeDefined()
  })

  it('renders description text about shortcuts', () => {
    render(<ShortcutLegendCard items={[]} />)
    expect(screen.getByText(/Shortcuts are disabled when focus is in input fields/)).toBeDefined()
  })
})
