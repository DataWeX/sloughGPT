import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { TypographyScale } from './TypographyScale'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
}))

afterEach(() => cleanup())

describe('TypographyScale', () => {
  const entries = [
    { role: 'Page Title', class: 'text-2xl font-semibold', sample: 'Noir Violet', font: 'font-rubik', weight: '600' },
    { role: 'Body', class: 'text-sm', sample: 'Warm, sophisticated.', font: 'font-rubik', weight: '400' },
    { role: 'Caption', class: 'text-xs text-muted-foreground', sample: 'Last updated', font: 'font-rubik', weight: '400' },
  ]

  it('renders all typography roles', () => {
    render(<TypographyScale entries={entries} />)
    expect(screen.getByText('Page Title')).toBeDefined()
    expect(screen.getByText('Body')).toBeDefined()
    expect(screen.getByText('Caption')).toBeDefined()
  })

  it('renders sample text for each entry', () => {
    render(<TypographyScale entries={entries} />)
    expect(screen.getByText('Noir Violet')).toBeDefined()
    expect(screen.getByText('Warm, sophisticated.')).toBeDefined()
    expect(screen.getByText('Last updated')).toBeDefined()
  })

  it('renders CSS class annotations', () => {
    render(<TypographyScale entries={entries} />)
    expect(screen.getByText('text-2xl font-semibold')).toBeDefined()
    expect(screen.getByText('text-sm')).toBeDefined()
  })

  it('renders a card wrapper', () => {
    render(<TypographyScale entries={entries} />)
    expect(screen.getAllByTestId('card').length).toBeGreaterThanOrEqual(1)
  })

  it('renders empty list gracefully', () => {
    render(<TypographyScale entries={[]} />)
    expect(screen.getAllByTestId('card').length).toBe(1)
  })
})
