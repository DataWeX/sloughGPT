import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconChat: () => <span data-testid="icon-chat" />,
  IconModels: () => <span data-testid="icon-models" />,
}))

vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual<any>('@sloughgpt/strui')
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return { ...actual, Card: passthrough, CardContent: passthrough, CardHeader: passthrough, CardTitle: passthrough, CardDescription: passthrough }
})

const t = (k: string) => k

import { StatsGrid } from './StatsGrid'

const online = { apiStatus: 'online', modelCount: 3, currentSoul: { name: 'Friendly', description: '', traits: [] }, modelStatus: { loaded: true, model: 'gpt2' }, inferenceCount: 42, t }

describe('StatsGrid', () => {
  it('shows Online when api is online', () => {
    render(<StatsGrid {...online} />)
    expect(screen.getByText('Online')).toBeDefined()
  })

  it('hides Online and shows skeleton when loading', () => {
    const { container } = render(<StatsGrid {...online} apiStatus="loading" />)
    expect(container.querySelector('.animate-pulse')).toBeTruthy()
  })

  it('shows model count', () => {
    render(<StatsGrid {...online} />)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
  })

  it('shows em dash when modelCount is null', () => {
    render(<StatsGrid {...online} modelCount={null} />)
    expect(screen.getByText('\u2014')).toBeDefined()
  })

  it('shows soul name', () => {
    render(<StatsGrid {...online} />)
    expect(screen.getAllByText('Friendly').length).toBeGreaterThanOrEqual(1)
  })

  it('shows model + soul when loaded', () => {
    render(<StatsGrid {...online} />)
    expect(screen.getAllByText(/gpt2 \+ Friendly/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows Not loaded when model not loaded', () => {
    render(<StatsGrid {...online} modelStatus={{ loaded: false, model: null }} />)
    expect(screen.getAllByText('Not loaded').length).toBeGreaterThanOrEqual(1)
  })

  it('shows conversations when inferenceCount provided', () => {
    render(<StatsGrid {...online} />)
    expect(screen.getAllByText(/42 conversations/).length).toBeGreaterThanOrEqual(1)
  })

  it('hides conversations when inferenceCount is null', () => {
    render(<StatsGrid {...online} inferenceCount={null} />)
    expect(screen.queryByText(/conversations/)).toBeNull()
  })
})
