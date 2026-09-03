import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { UsageStats } from './UsageStats'

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))

vi.mock('@/lib/format-bytes', () => ({
  formatBytes: (bytes: number) => `${(bytes / 1024).toFixed(1)} KB`,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
  }
})

const baseProps = {
  apiStatus: 'online' as string,
  convStats: { totalConversations: 15, totalMessages: 320, totalWords: 8500, activeDays: 5, mostActiveHour: 14 },
  datasetStats: { totalDatasets: 3, totalSize: 10240, totalSamples: 1500 },
}

describe('UsageStats', () => {
  it('renders nothing when offline', () => {
    const { container } = render(<UsageStats {...baseProps} apiStatus="offline" />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when both stats are null', () => {
    const { container } = render(<UsageStats {...baseProps} convStats={null} datasetStats={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows conversation stats', () => {
    const { container } = render(<UsageStats {...baseProps} />)
    expect(container.textContent).toContain('15')
    expect(container.textContent).toContain('320')
    expect(container.textContent).toContain('8,500')
    expect(container.textContent).toContain('5')
  })

  it('shows most active hour', () => {
    const { container } = render(<UsageStats {...baseProps} />)
    expect(container.textContent).toContain('Most active at 14:00')
  })

  it('hides most active hour when null', () => {
    const { container } = render(<UsageStats {...baseProps} convStats={{ ...baseProps.convStats, mostActiveHour: null }} />)
    expect(container.textContent).not.toMatch(/Most active/)
  })

  it('shows dataset stats', () => {
    const { container } = render(<UsageStats {...baseProps} />)
    expect(container.textContent).toContain('10.0 KB')
    expect(container.textContent).toContain('1,500')
  })

  it('links to datasets page', () => {
    const { container } = render(<UsageStats {...baseProps} />)
    const links = container.querySelectorAll('a')
    const datasetsLink = Array.from(links).find(a => a.getAttribute('href') === '/datasets')
    expect(datasetsLink).toBeDefined()
    expect(datasetsLink!.textContent).toContain('View all →')
  })

  it('hides conversation card when no conversations', () => {
    const { container } = render(<UsageStats {...baseProps} convStats={{ ...baseProps.convStats, totalConversations: 0 }} />)
    expect(container.textContent).not.toContain('Your stats')
  })

  it('hides dataset card when no datasets', () => {
    const { container } = render(<UsageStats {...baseProps} datasetStats={{ ...baseProps.datasetStats, totalDatasets: 0 }} />)
    expect(container.querySelectorAll('a[href="/datasets"]').length).toBe(0)
  })
})
