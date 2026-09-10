/// <reference types="vitest" />
// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { VMTraceViewer } from './VMTraceViewer'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

describe('VMTraceViewer', () => {
  const trace = [
    { step: 0, eip: '0x00001000', opcode: 'MOV', operands: 'EAX, 42' },
    { step: 1, eip: '0x00001003', opcode: 'HLT', operands: '' },
  ]

  it('renders nothing when trace is empty', () => {
    const { container } = render(<VMTraceViewer trace={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders the title with trace length', () => {
    render(<VMTraceViewer trace={trace} />)
    expect(screen.getByText('Execution Trace (first 2 steps)')).toBeDefined()
  })

  it('renders table headers', () => {
    render(<VMTraceViewer trace={trace} />)
    expect(screen.getByText('#')).toBeDefined()
    expect(screen.getByText('EIP')).toBeDefined()
    expect(screen.getByText('Opcode')).toBeDefined()
    expect(screen.getByText('Operands')).toBeDefined()
  })

  it('renders trace rows', () => {
    render(<VMTraceViewer trace={trace} />)
    expect(screen.getByText('0x00001000')).toBeDefined()
    expect(screen.getByText('MOV')).toBeDefined()
    expect(screen.getByText('EAX, 42')).toBeDefined()
    expect(screen.getByText('0x00001003')).toBeDefined()
    expect(screen.getByText('HLT')).toBeDefined()
  })

  it('renders step numbers', () => {
    render(<VMTraceViewer trace={trace} />)
    expect(screen.getByText('0')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
  })

  it('renders multiple trace entries', () => {
    const longTrace = [
      { step: 0, eip: '0x1000', opcode: 'MOV', operands: 'EAX, 1' },
      { step: 1, eip: '0x1003', opcode: 'ADD', operands: 'EAX, 2' },
      { step: 2, eip: '0x1006', opcode: 'HLT', operands: '' },
    ]
    render(<VMTraceViewer trace={longTrace} />)
    expect(screen.getByText('Execution Trace (first 3 steps)')).toBeDefined()
    expect(screen.getByText('ADD')).toBeDefined()
  })
})
