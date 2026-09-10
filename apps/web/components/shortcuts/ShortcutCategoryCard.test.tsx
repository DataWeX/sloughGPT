/// <reference types="vitest" />
import { render, screen, cleanup, within } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { ShortcutCategoryCard, type Shortcut } from './ShortcutCategoryCard'
import { Keyboard, RefreshCw } from 'lucide-react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const shortcuts: Shortcut[] = [
  { keys: ['Ctrl', 'K'], label: 'Command palette', icon: Keyboard },
  { keys: ['R'], label: 'Refresh data', icon: RefreshCw },
]

describe('ShortcutCategoryCard', () => {
  it('renders the category title', () => {
    render(<ShortcutCategoryCard title="Global" shortcuts={[]} />)
    expect(screen.getByText('Global')).toBeDefined()
  })

  it('renders shortcut labels', () => {
    render(<ShortcutCategoryCard title="Global" shortcuts={shortcuts} />)
    expect(screen.getByText('Command palette')).toBeDefined()
    expect(screen.getByText('Refresh data')).toBeDefined()
  })

  it('renders kbd keys for each shortcut', () => {
    render(<ShortcutCategoryCard title="Global" shortcuts={shortcuts} />)
    const kbds = screen.getAllByText('Ctrl')
    expect(kbds.length).toBeGreaterThanOrEqual(1)
  })

  it('renders with empty shortcuts array', () => {
    render(<ShortcutCategoryCard title="Empty" shortcuts={[]} />)
    expect(screen.getByText('Empty')).toBeDefined()
    const content = screen.getByTestId('card-content')
    expect(within(content).queryByText('+')).toBeNull()
  })

  it('renders plus separator between multi-key combos', () => {
    render(<ShortcutCategoryCard title="Test" shortcuts={shortcuts} />)
    const separators = screen.getAllByText('+')
    expect(separators.length).toBeGreaterThanOrEqual(1)
  })
})
