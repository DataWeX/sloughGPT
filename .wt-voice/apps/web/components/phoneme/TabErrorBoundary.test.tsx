import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div data-testid="card-header">{children}</div>,
  CardTitle: ({ children, ...props }: any) => <h2 data-testid="card-title" {...props}>{children}</h2>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
}))

import TabErrorBoundary from './TabErrorBoundary'

afterEach(cleanup)

let shouldThrow = true
function ConditionalBomb() {
  if (shouldThrow) throw new Error('test crash')
  return <p>recovered</p>
}

describe('TabErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <TabErrorBoundary tabName="Chat">
        <p>child content</p>
      </TabErrorBoundary>
    )
    expect(screen.getByText('child content')).toBeInTheDocument()
  })

  it('catches render errors and shows fallback UI', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    shouldThrow = true
    render(
      <TabErrorBoundary tabName="Chat">
        <ConditionalBomb />
      </TabErrorBoundary>
    )
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText(/The Chat tab encountered an error/)).toBeInTheDocument()
    expect(screen.getByText('test crash')).toBeInTheDocument()
    spy.mockRestore()
  })

  it('resets error state when Try again is clicked', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    shouldThrow = true
    render(
      <TabErrorBoundary tabName="Chat">
        <ConditionalBomb />
      </TabErrorBoundary>
    )
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    shouldThrow = false
    fireEvent.click(screen.getByText('Try again'))
    expect(screen.getByText('recovered')).toBeInTheDocument()
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
    spy.mockRestore()
  })
})
