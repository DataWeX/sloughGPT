import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { SystemHealth } from './SystemHealth'

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))

vi.mock('@/lib/chat-utils', () => ({
  formatUptime: (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    return h > 0 ? `${h}h ${m}m` : `${m}m`
  },
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children, className }: any) => <div className={className}>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    cn: (...args: any[]) => args.filter(Boolean).join(' '),
  
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
  liveHealth: {
    cpu_percent: 45.2,
    memory_percent: 67.8,
    request_count: 1234,
    uptime_seconds: 7200,
  } as any,
}

describe('SystemHealth', () => {
  it('shows skeleton when loading', () => {
    const { container } = render(<SystemHealth {...baseProps} loading={true} />)
    expect(container.querySelector('.animate-pulse')).toBeTruthy()
  })

  it('renders nothing when offline', () => {
    const { container } = render(<SystemHealth {...baseProps} apiStatus="offline" />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when liveHealth is null', () => {
    const { container } = render(<SystemHealth {...baseProps} liveHealth={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows CPU percentage', () => {
    const { container } = render(<SystemHealth {...baseProps} />)
    expect(container.textContent).toContain('45%')
  })

  it('shows memory percentage', () => {
    const { container } = render(<SystemHealth {...baseProps} />)
    expect(container.textContent).toContain('68%')
  })

  it('shows request count', () => {
    const { container } = render(<SystemHealth {...baseProps} />)
    expect(container.textContent).toContain('1,234')
  })

  it('shows uptime', () => {
    const { container } = render(<SystemHealth {...baseProps} />)
    expect(container.textContent).toContain('2h 0m')
  })

  it('shows em dash for null CPU', () => {
    const { container } = render(<SystemHealth {...baseProps} liveHealth={{ ...baseProps.liveHealth, cpu_percent: null }} />)
    expect(container.textContent).toContain('\u2014')
  })

  it('shows em dash for null memory', () => {
    const { container } = render(<SystemHealth {...baseProps} liveHealth={{ ...baseProps.liveHealth, memory_percent: null }} />)
    expect(container.textContent).toContain('\u2014')
  })

  it('shows em dash for 0 uptime', () => {
    const { container } = render(<SystemHealth {...baseProps} liveHealth={{ ...baseProps.liveHealth, uptime_seconds: 0 }} />)
    expect(container.textContent).toContain('\u2014')
  })

  it('links to monitoring page', () => {
    const { container } = render(<SystemHealth {...baseProps} />)
    const links = container.querySelectorAll('a')
    const monitoringLink = Array.from(links).find(a => a.getAttribute('href') === '/monitoring')
    expect(monitoringLink).toBeDefined()
    expect(monitoringLink!.textContent).toContain('Details →')
  })

  it('shows how it works section', () => {
    const { container } = render(<SystemHealth {...baseProps} />)
    expect(container.textContent).toContain('How it works')
    expect(container.textContent).toMatch(/Mix and match AI models/)
  })
})
