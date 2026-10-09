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
  
Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}
})

const baseProps = {
  apiStatus: 'online' as string,
  loading: false,
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
