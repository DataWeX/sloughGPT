/**
 */
import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { StatCard, KpiGrid, EmptyCard, Skeleton, LoadingDots, ListRow, ListSection } from './display'

describe('StatCard', () => {
  it('renders label and value', () => {
    render(<StatCard label="Users" value={42} />)
    expect(screen.getByText('Users')).toBeInTheDocument()
    expect(screen.getByText('42')).toBeInTheDocument()
  })

  it('renders trend with positive sign', () => {
    render(<StatCard label="Revenue" value="$1k" trend={{ value: 12, positive: true }} />)
    expect(screen.getByText('+12%')).toBeInTheDocument()
  })

  it('renders trend without positive sign when negative', () => {
    render(<StatCard label="Errors" value="5" trend={{ value: 8 }} />)
    expect(screen.getByText('8%')).toBeInTheDocument()
  })

  it('renders icon when provided', () => {
    render(<StatCard label="Test" value={1} icon={<span data-testid="test-icon" />} />)
    expect(screen.getByTestId('test-icon')).toBeInTheDocument()
  })

  it('does not render trend section when absent', () => {
    const { container } = render(<StatCard label="Plain" value={0} />)
    expect(container.querySelector('.text-success')).not.toBeInTheDocument()
    expect(container.querySelector('.text-destructive')).not.toBeInTheDocument()
  })
})

describe('KpiGrid', () => {
  it('renders children in grid layout', () => {
    render(
      <KpiGrid>
        <div data-testid="child-1" />
        <div data-testid="child-2" />
      </KpiGrid>
    )
    expect(screen.getByTestId('child-1')).toBeInTheDocument()
    expect(screen.getByTestId('child-2')).toBeInTheDocument()
  })

  it('applies column class for 2 columns', () => {
    const { container } = render(<KpiGrid columns={2}><div /></KpiGrid>)
    expect(container.firstChild).toHaveClass('grid-cols-2')
  })

  it('defaults to 4 columns', () => {
    const { container } = render(<KpiGrid><div /></KpiGrid>)
    expect(container.firstChild).toHaveClass('grid-cols-2')
    expect(container.firstChild).toHaveClass('sm:grid-cols-4')
  })
})

describe('EmptyCard', () => {
  it('renders default message', () => {
    render(<EmptyCard />)
    expect(screen.getByText('No items')).toBeInTheDocument()
  })

  it('renders custom message', () => {
    render(<EmptyCard message="Nothing here" />)
    expect(screen.getByText('Nothing here')).toBeInTheDocument()
  })

  it('renders action when provided', () => {
    render(<EmptyCard action={<button>Add</button>} />)
    expect(screen.getByText('Add')).toBeInTheDocument()
  })

  it('does not render action div when absent', () => {
    const { container } = render(<EmptyCard />)
    expect(container.querySelector('.mt-3')).not.toBeInTheDocument()
  })
})

describe('Skeleton', () => {
  it('renders with animate-pulse class', () => {
    const { container } = render(<Skeleton />)
    expect(container.firstChild).toHaveClass('animate-pulse')
    expect(container.firstChild).toHaveClass('rounded-md')
    expect(container.firstChild).toHaveClass('bg-muted')
  })

  it('applies custom className', () => {
    const { container } = render(<Skeleton className="h-10 w-full" />)
    expect(container.firstChild).toHaveClass('h-10')
    expect(container.firstChild).toHaveClass('w-full')
  })
})

describe('LoadingDots', () => {
  it('renders three dots', () => {
    const { container } = render(<LoadingDots />)
    expect(container.querySelectorAll('.animate-bounce')).toHaveLength(3)
  })

  it('applies custom className', () => {
    const { container } = render(<LoadingDots className="ml-2" />)
    expect(container.firstChild).toHaveClass('ml-2')
  })
})

describe('ListRow', () => {
  it('renders label', () => {
    render(<ListRow label="Name" />)
    expect(screen.getByText('Name')).toBeInTheDocument()
  })

  it('renders value when provided', () => {
    render(<ListRow label="Status" value="Active" />)
    expect(screen.getByText('Active')).toBeInTheDocument()
  })

  it('does not render value element when absent', () => {
    const { container } = render(<ListRow label="No val" />)
    expect(container.querySelector('.text-muted-foreground')).not.toBeInTheDocument()
  })

  it('renders action node when provided', () => {
    render(<ListRow label="Item" action={<button data-testid="action-btn" />} />)
    expect(screen.getByTestId('action-btn')).toBeInTheDocument()
  })
})

describe('ListSection', () => {
  it('renders title when provided', () => {
    render(<ListSection title="Section A"><div /></ListSection>)
    expect(screen.getByText('Section A')).toBeInTheDocument()
  })

  it('does not render title when absent', () => {
    const { container } = render(<ListSection><div /></ListSection>)
    expect(container.querySelector('.text-\\[10px\\]')).not.toBeInTheDocument()
  })

  it('renders children', () => {
    render(<ListSection><div data-testid="child" /></ListSection>)
    expect(screen.getByTestId('child')).toBeInTheDocument()
  })

  it('has border rounded wrapper', () => {
    const { container } = render(<ListSection><div /></ListSection>)
    expect(container.querySelector('.border.rounded-lg')).toBeInTheDocument()
  })
})
