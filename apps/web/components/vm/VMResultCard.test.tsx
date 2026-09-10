/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { VMResultCard } from './VMResultCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Badge: ({ children, ...props }: any) => <span data-testid="badge" {...props}>{children}</span>,
}))

describe('VMResultCard', () => {
  const mockResult = {
    success: true,
    exit_code: 0,
    steps_executed: 42,
    elapsed_ms: 1.5,
    status: 'halted',
    registers: [
      { name: 'EAX', value: 42, hex: '0x0000002A' },
      { name: 'EBX', value: 0, hex: '0x00000000' },
    ],
    eip_hex: '0x00001005',
  }

  it('renders the title', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('Result')).toBeDefined()
  })

  it('renders nothing when result is null', () => {
    const { container } = render(<VMResultCard result={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders status badge', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('halted')).toBeDefined()
  })

  it('renders exit code', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('0x0')).toBeDefined()
  })

  it('renders steps executed', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('42')).toBeDefined()
  })

  it('renders elapsed time', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('1.5ms')).toBeDefined()
  })

  it('renders registers', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('EAX')).toBeDefined()
    expect(screen.getByText('0x0000002A')).toBeDefined()
  })

  it('renders error message when present', () => {
    const errorResult = { ...mockResult, error: 'Division by zero', success: false }
    render(<VMResultCard result={errorResult} />)
    expect(screen.getByText('Division by zero')).toBeDefined()
  })

  it('renders EIP', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('0x00001005')).toBeDefined()
  })
})
