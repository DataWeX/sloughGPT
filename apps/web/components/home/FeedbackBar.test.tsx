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
    expect(container.textContent).toContain('80%')
  })

  it('shows warning ratio when < 50%', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={makeStats({ ratio: 0.3 })} />)
    expect(container.textContent).toContain('30%')
  })

  it('links to training page', () => {
    const { container } = render(<FeedbackBar loading={false} feedbackStats={makeStats()} />)
    const links = container.querySelectorAll('a')
    const trainingLink = Array.from(links).find(a => a.getAttribute('href') === '/training')
    expect(trainingLink).toBeDefined()
    expect(trainingLink!.textContent).toContain('Train from feedback →')
  })
})
