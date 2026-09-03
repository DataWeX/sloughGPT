import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { NavigationGrid } from './NavigationGrid'

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconChat: () => <span data-testid="icon-chat" />,
  IconModels: () => <span data-testid="icon-models" />,
}))

vi.mock('@sloughgpt/strui', async () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    IconChevronRight: () => <span data-testid="chevron" />,
    IconSearch: () => <span data-testid="icon-search" />,
    IconBolt: () => <span data-testid="icon-bolt" />,
    IconChart: () => <span data-testid="icon-chart" />,
    cn: (...args: any[]) => args.filter(Boolean).join(' '),
  }
})

vi.mock('@/lib/format-bytes', () => ({
  formatBytes: (bytes: number) => `${bytes} B`,
}))

const baseProps = {
  apiStatus: 'online' as string,
  modelStatus: { loaded: true, model: 'gpt2' },
  datasetStats: { totalDatasets: 5, totalSize: 1024, totalSamples: 100 },
}

describe('NavigationGrid', () => {
  it('renders all 6 navigation tiles', () => {
    const { container } = render(<NavigationGrid {...baseProps} />)
    expect(container.textContent).toContain('Start chatting')
    expect(container.textContent).toContain('Personalities')
    expect(container.textContent).toContain('Datasets')
    expect(container.textContent).toContain('Teach me')
    expect(container.textContent).toContain('System Health')
    expect(container.textContent).toContain('Knowledge')
  })

  it('links to correct routes', () => {
    const { container } = render(<NavigationGrid {...baseProps} />)
    const links = container.querySelectorAll('a')
    const hrefs = Array.from(links).map(a => a.getAttribute('href'))
    expect(hrefs).toContain('/chat')
    expect(hrefs).toContain('/models')
    expect(hrefs).toContain('/datasets')
    expect(hrefs).toContain('/training')
    expect(hrefs).toContain('/monitoring')
    expect(hrefs).toContain('/knowledge')
  })

  it('shows dataset count badge on Datasets tile', () => {
    const { container } = render(<NavigationGrid {...baseProps} />)
    const links = container.querySelectorAll('a')
    const datasetsLink = Array.from(links).find(a => a.getAttribute('href') === '/datasets')
    expect(datasetsLink!.textContent).toContain('5')
  })

  it('hides dataset badge when 0 datasets', () => {
    const { container } = render(<NavigationGrid {...baseProps} datasetStats={{ totalDatasets: 0, totalSize: 0, totalSamples: 0 }} />)
    const links = container.querySelectorAll('a')
    const datasetsLink = Array.from(links).find(a => a.getAttribute('href') === '/datasets')
    expect(datasetsLink!.textContent).not.toMatch(/\b0\b/)
  })

  it('hides dataset badge when stats are null', () => {
    const { container } = render(<NavigationGrid {...baseProps} datasetStats={null} />)
    const links = container.querySelectorAll('a')
    const datasetsLink = Array.from(links).find(a => a.getAttribute('href') === '/datasets')
    const badge = datasetsLink!.querySelector('span.text-xs')
    expect(badge).toBeNull()
  })

  it('shows subtitles on desktop', () => {
    const { container } = render(<NavigationGrid {...baseProps} />)
    expect(container.textContent).toContain('Ask anything, get answers')
    expect(container.textContent).toContain("Switch your agent's personality")
    expect(container.textContent).toContain('Manage training data')
    expect(container.textContent).toContain('Train from your writing')
    expect(container.textContent).toContain('Monitor API and resources')
    expect(container.textContent).toContain('Facts the AI remembers')
  })
})
