import { describe, it, expect, vi } from 'vitest'
import { render, fireEvent } from '@testing-library/react'
import { RecentActivity } from './RecentActivity'

const mockPush = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))

vi.mock('@/lib/time-ago', () => ({
  timeAgo: () => '2 hours ago',
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    cn: (...args: any[]) => args.filter(Boolean).join(' '),
  }
})

const baseProps = {
  apiStatus: 'online' as string,
  modelStatus: { loaded: true, model: 'gpt2' },
  recentSessions: [
    { id: 's1', name: 'Chat 1', updated_at: '2026-01-01', message_count: 10, starred: true, pinned: false },
    { id: 's2', name: 'Chat 2', updated_at: '2026-01-02', message_count: 5 },
  ],
  recentJobs: [
    { id: 'j1', name: 'Training run', status: 'completed', created_at: '2026-01-01' },
    { id: 'j2', name: 'Another run', status: 'running', created_at: '2026-01-02' },
  ],
  recentDatasets: [
    { id: 'd1', name: 'Dataset A', samples: 1000 },
    { id: 'd2', name: 'Dataset B', samples: 500 },
  ],
}

describe('RecentActivity', () => {
  it('renders nothing when offline', () => {
    const { container } = render(<RecentActivity {...baseProps} apiStatus="offline" />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when model not loaded', () => {
    const { container } = render(<RecentActivity {...baseProps} modelStatus={{ loaded: false, model: null }} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows recent sessions', () => {
    const { container } = render(<RecentActivity {...baseProps} />)
    expect(container.textContent).toContain('Chat 1')
    expect(container.textContent).toContain('Chat 2')
  })

  it('shows starred icon for starred sessions', () => {
    const { container } = render(<RecentActivity {...baseProps} />)
    expect(container.textContent).toContain('\u2605')
  })

  it('navigates to chat on session click', () => {
    const { container } = render(<RecentActivity {...baseProps} />)
    const buttons = container.querySelectorAll('button')
    const chat1Btn = Array.from(buttons).find(b => b.textContent?.includes('Chat 1'))
    fireEvent.click(chat1Btn!)
    expect(mockPush).toHaveBeenCalledWith('/chat?session=s1')
  })

  it('shows recent training jobs', () => {
    const { container } = render(<RecentActivity {...baseProps} />)
    expect(container.textContent).toContain('Training run')
    expect(container.textContent).toContain('completed')
    expect(container.textContent).toContain('running')
  })

  it('shows recent datasets card when datasets exist', () => {
    const { container } = render(<RecentActivity {...baseProps} />)
    expect(container.textContent).toContain('Recent datasets')
    expect(container.textContent).toContain('Dataset A')
  })

  it('navigates to training on dataset click', () => {
    const { container } = render(<RecentActivity {...baseProps} />)
    const buttons = container.querySelectorAll('button')
    const datasetBtn = Array.from(buttons).find(b => b.textContent?.includes('Dataset A'))
    fireEvent.click(datasetBtn!)
    expect(mockPush).toHaveBeenCalledWith('/training?dataset=d1')
  })

  it('hides datasets card when empty', () => {
    const { container } = render(<RecentActivity {...baseProps} recentDatasets={[]} />)
    expect(container.textContent).not.toContain('Recent datasets')
  })

  it('links to datasets page', () => {
    const { container } = render(<RecentActivity {...baseProps} />)
    const links = container.querySelectorAll('a')
    const datasetsLink = Array.from(links).find(a => a.getAttribute('href') === '/datasets')
    expect(datasetsLink).toBeDefined()
    expect(datasetsLink!.textContent).toContain('View all →')
  })
})
