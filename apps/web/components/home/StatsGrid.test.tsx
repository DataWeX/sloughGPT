import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { StatsGrid } from './StatsGrid'

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconChat: () => <span data-testid="icon-chat" />,
  IconModels: () => <span data-testid="icon-models" />,
}))

const t = (k: string) => k

const base = { apiStatus: 'online', modelCount: 3, currentSoul: { name: 'Friendly', description: '', traits: [] }, modelStatus: { loaded: true, model: 'gpt2' }, inferenceCount: 42, t }

describe('StatsGrid', () => {
  it('shows skeleton divs when loading', () => { const { container } = render(<StatsGrid {...base} apiStatus="loading" />); expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThanOrEqual(1) })
  it('shows model count', () => { render(<StatsGrid {...base} />); expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1) })
  it('shows em dash when null', () => { render(<StatsGrid {...base} modelCount={null} />); expect(screen.getByText('\u2014')).toBeDefined() })
  it('shows soul name', () => { render(<StatsGrid {...base} />); expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1) })
  it('shows model + soul', () => { render(<StatsGrid {...base} />); expect(screen.getAllByText(/gpt2 \+ Friendly/).length).toBeGreaterThanOrEqual(1) })
  it('shows Not loaded', () => { render(<StatsGrid {...base} modelStatus={{ loaded: false, model: null }} />); expect(screen.getAllByText('Not loaded').length).toBeGreaterThanOrEqual(1) })
  it('shows conversations', () => { render(<StatsGrid {...base} />); expect(screen.getAllByText(/42 conversations/).length).toBeGreaterThanOrEqual(1) })
  it('hides conversations when null', () => { const { container } = render(<StatsGrid {...base} inferenceCount={null} />); expect(container.textContent).not.toMatch(/\d+ conversations/) })
})
