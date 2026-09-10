/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { DashboardHealthStatus } from './DashboardHealthStatus'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
}))

describe('DashboardHealthStatus', () => {
  it('renders nothing when status is null', () => {
    const { container } = render(<DashboardHealthStatus status={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders healthy status', () => {
    render(<DashboardHealthStatus status="healthy" />)
    expect(screen.getByText('Workspace healthy')).toBeDefined()
  })

  it('renders error status', () => {
    render(<DashboardHealthStatus status="error" />)
    expect(screen.getByText('Workspace error')).toBeDefined()
  })

  it('renders warning status', () => {
    render(<DashboardHealthStatus status="warning" />)
    expect(screen.getByText('Workspace warning')).toBeDefined()
  })

  it('renders re-check button when onRecheck provided', () => {
    const onRecheck = vi.fn()
    render(<DashboardHealthStatus status="healthy" onRecheck={onRecheck} />)
    fireEvent.click(screen.getByText('Re-check'))
    expect(onRecheck).toHaveBeenCalledOnce()
  })

  it('shows checking state', () => {
    render(<DashboardHealthStatus status="healthy" checking onRecheck={vi.fn()} />)
    expect(screen.getByText('Checking...')).toBeDefined()
  })
})
