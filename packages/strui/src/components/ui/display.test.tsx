import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { EmptyCard, KpiGrid, ListRow, ListSection, LoadingDots, Skeleton, StatCard } from './display'

afterEach(() => {
  cleanup()
})

describe('StatCard', () => {
  it('renders label and value', () => {
    const html = renderToStaticMarkup(<StatCard label="Revenue" value="$1,234" />)
    expect(html).toContain('Revenue')
    expect(html).toContain('$1,234')
  })

  it('renders icon', () => {
    const html = renderToStaticMarkup(<StatCard label="Revenue" value="5" icon={<span data-icon="dollar">$</span>} />)
    expect(html).toContain('data-icon="dollar"')
  })

  it('renders a positive trend with up arrow and success styling', () => {
    const html = renderToStaticMarkup(
      <StatCard label="Revenue" value="5" trend={{ value: 12, positive: true, label: 'vs last week' }} />,
    )
    expect(html).toContain('↑ 12%')
    expect(html).toContain('text-success')
    expect(html).toContain('vs last week')
  })

  it('renders a negative trend with down arrow and destructive styling', () => {
    const html = renderToStaticMarkup(<StatCard label="Revenue" value="5" trend={{ value: -12, positive: false }} />)
    expect(html).toContain('↓ 12%')
    expect(html).toContain('text-destructive')
  })

  it('renders description', () => {
    const html = renderToStaticMarkup(<StatCard label="Revenue" value="5" description="Last 30 days" />)
    expect(html).toContain('Last 30 days')
  })

  it('prefers trend.label over description', () => {
    const html = renderToStaticMarkup(
      <StatCard label="Revenue" value="5" description="Description" trend={{ value: 1, label: 'Trend label' }} />,
    )
    expect(html).toContain('Trend label')
    expect(html).not.toContain('Description')
  })

  it('renders a skeleton instead of content when loading', () => {
    const html = renderToStaticMarkup(<StatCard label="Revenue" value="$1,234" loading />)
    expect(html).toContain('animate-pulse')
    expect(html).not.toContain('Revenue')
    expect(html).not.toContain('$1,234')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<StatCard label="Revenue" value="5" className="my-card" />)
    expect(html).toContain('my-card')
  })
})

describe('KpiGrid', () => {
  it('defaults to 4 columns', () => {
    const html = renderToStaticMarkup(<KpiGrid><span>a</span></KpiGrid>)
    expect(html).toContain('grid-cols-2 sm:grid-cols-4')
  })

  it('renders children', () => {
    const html = renderToStaticMarkup(
      <KpiGrid columns={2}>
        <span data-kpi="a">A</span>
        <span data-kpi="b">B</span>
      </KpiGrid>,
    )
    expect(html).toContain('data-kpi="a"')
    expect(html).toContain('data-kpi="b"')
  })

  it('maps column counts to grid classes', () => {
    expect(renderToStaticMarkup(<KpiGrid columns={1}><span /></KpiGrid>)).toContain('grid-cols-1')
    expect(renderToStaticMarkup(<KpiGrid columns={3}><span /></KpiGrid>)).toContain('grid-cols-1 sm:grid-cols-3')
    expect(renderToStaticMarkup(<KpiGrid columns={5}><span /></KpiGrid>)).toContain(
      'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5',
    )
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<KpiGrid className="my-grid"><span /></KpiGrid>)
    expect(html).toContain('my-grid')
  })
})

describe('ListRow', () => {
  it('renders label and value', () => {
    const html = renderToStaticMarkup(<ListRow label="Name" value="Alice" />)
    expect(html).toContain('Name')
    expect(html).toContain('Alice')
  })

  it('renders icon and action', () => {
    const html = renderToStaticMarkup(
      <ListRow label="Name" icon={<span data-icon="u">U</span>} action={<span data-action="del">X</span>} />,
    )
    expect(html).toContain('data-icon="u"')
    expect(html).toContain('data-action="del"')
  })

  it('renders a button when onClick is provided', () => {
    const html = renderToStaticMarkup(<ListRow label="Name" onClick={() => {}} />)
    expect(html).toContain('<button')
    expect(html).toContain('type="button"')
  })

  it('renders a div, not a button, without onClick', () => {
    const html = renderToStaticMarkup(<ListRow label="Name" />)
    expect(html).not.toContain('<button')
  })

  it('fires onClick on click', () => {
    const onClick = vi.fn()
    render(<ListRow label="Name" onClick={onClick} />)
    fireEvent.click(screen.getByText('Name'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<ListRow label="Name" className="my-row" />)
    expect(html).toContain('my-row')
  })
})

describe('ListSection', () => {
  it('renders title and description', () => {
    const html = renderToStaticMarkup(<ListSection title="Agents" description="All agents"><span>x</span></ListSection>)
    expect(html).toContain('Agents')
    expect(html).toContain('All agents')
  })

  it('renders children', () => {
    const html = renderToStaticMarkup(<ListSection><span data-child="row">Row</span></ListSection>)
    expect(html).toContain('data-child="row"')
  })

  it('omits the header when there is no title or description', () => {
    const html = renderToStaticMarkup(<ListSection><span>Row</span></ListSection>)
    expect(html).not.toContain('text-[10px]')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<ListSection className="my-section"><span /></ListSection>)
    expect(html).toContain('my-section')
  })
})

describe('EmptyCard', () => {
  it('defaults the message to No items', () => {
    const html = renderToStaticMarkup(<EmptyCard />)
    expect(html).toContain('No items')
  })

  it('renders a custom message and description', () => {
    const html = renderToStaticMarkup(<EmptyCard message="Nothing here" description="Add one to get started" />)
    expect(html).toContain('Nothing here')
    expect(html).toContain('Add one to get started')
  })

  it('renders an icon', () => {
    const html = renderToStaticMarkup(<EmptyCard icon={<span data-icon="box">B</span>} />)
    expect(html).toContain('data-icon="box"')
  })

  it('renders an action', () => {
    const html = renderToStaticMarkup(<EmptyCard action={<button data-action="create">Create</button>} />)
    expect(html).toContain('data-action="create"')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<EmptyCard className="my-empty" />)
    expect(html).toContain('my-empty')
  })
})

describe('Skeleton', () => {
  it('renders a single block by default', () => {
    const html = renderToStaticMarkup(<Skeleton />)
    expect(html).toContain('animate-pulse')
    expect(html).toContain('rounded-md')
  })

  it('renders N blocks when lines is provided', () => {
    const html = renderToStaticMarkup(<Skeleton lines={3} />)
    const matches = html.match(/animate-pulse/g)
    expect(matches?.length).toBe(3)
  })

  it('fills all but the last line to full width', () => {
    const html = renderToStaticMarkup(<Skeleton lines={3} />)
    expect(html).toContain('width:100%')
    expect(html).toContain('width:60%')
  })

  it('applies lastLineWidth to the final line', () => {
    const html = renderToStaticMarkup(<Skeleton lines={2} lastLineWidth="40%" />)
    expect(html).toContain('width:40%')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Skeleton className="my-skeleton" />)
    expect(html).toContain('my-skeleton')
  })
})

describe('LoadingDots', () => {
  it('renders a status with a Loading label', () => {
    const html = renderToStaticMarkup(<LoadingDots />)
    expect(html).toContain('role="status"')
    expect(html).toContain('aria-label="Loading"')
  })

  it('uses default size classes', () => {
    const html = renderToStaticMarkup(<LoadingDots />)
    expect(html).toContain('w-1.5 h-1.5')
  })

  it('supports sm and lg sizes', () => {
    expect(renderToStaticMarkup(<LoadingDots size="sm" />)).toContain('w-1 h-1')
    expect(renderToStaticMarkup(<LoadingDots size="lg" />)).toContain('w-2 h-2')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<LoadingDots className="my-dots" />)
    expect(html).toContain('my-dots')
  })
})
