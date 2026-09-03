import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { FeedbackBar } from './FeedbackBar'

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    IconThumbUp: () => <span data-testid="thumb-up" />,
    IconThumbDown: () => <span data-testid="thumb-down" />,
    cn: (...args: any[]) => args.filter(Boolean).join(' '),
  }
})

function makeStats(overrides: Record<string, any> = {}) {
  return {
    db_stats: {
      feedback_total: 100,
      thumbs_up: 80,
      thumbs_down: 20,
      ratio: 0.8,
      ...overrides,
    },
  } as any
}

describe('FeedbackBar', () => {
  it('shows skeleton when loading', () => {
    const { container } = render(<FeedbackBar loading feedbackStats={null as any} />)
    expect(container.querySelector('.animate-pulse')).toBeTruthy()
  })

  it('renders nothing when feedback_total is 0', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={makeStats({ feedback_total: 0 })} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when db_stats is missing', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={null as any} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows feedback count', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={makeStats()} />)
    expect(container.textContent).toContain('100')
  })

  it('shows thumbs up and down counts', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={makeStats()} />)
    expect(container.textContent).toContain('80')
    expect(container.textContent).toContain('20')
  })

  it('shows positive ratio when >= 50%', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={makeStats({ ratio: 0.8 })} />)
    expect(container.textContent).toContain('80% positive')
  })

  it('shows warning ratio when < 50%', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={makeStats({ ratio: 0.3 })} />)
    expect(container.textContent).toContain('30% positive')
  })

  it('links to training page', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={makeStats()} />)
    const links = container.querySelectorAll('a')
    const trainingLink = Array.from(links).find(a => a.getAttribute('href') === '/training')
    expect(trainingLink).toBeDefined()
    expect(trainingLink!.textContent).toContain('Train from feedback →')
  })
})
