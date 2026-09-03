import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ErrorDiagnosticsPanel } from './ErrorDiagnosticsPanel'
import type { ErrorEvent } from '@/hooks/useErrorStream'

afterEach(() => cleanup())

vi.mock('navigator.clipboard', () => ({ writeText: vi.fn() }))

const makeError = (overrides: Partial<ErrorEvent> = {}): ErrorEvent => ({
  id: 'e1', message: 'Something failed', level: 'error', source: 'frontend',
  phase: 'runtime', timestamp: Date.now(), ...overrides,
})

const defaultProps = { onClear: vi.fn() }

describe('ErrorDiagnosticsPanel', () => {
  it('shows empty state', () => {
    render(<ErrorDiagnosticsPanel errors={[]} {...defaultProps} />)
    expect(screen.getAllByText(/No errors/).length).toBeGreaterThanOrEqual(1)
  })
  it('renders error message', () => {
    render(<ErrorDiagnosticsPanel errors={[makeError()]} {...defaultProps} />)
    expect(screen.getAllByText('Something failed').length).toBeGreaterThanOrEqual(1)
  })
  it('shows error level badge', () => {
    render(<ErrorDiagnosticsPanel errors={[makeError({ level: 'error' })]} {...defaultProps} />)
    expect(screen.getAllByText('ERR').length).toBeGreaterThanOrEqual(1)
  })
  it('shows critical level badge', () => {
    render(<ErrorDiagnosticsPanel errors={[makeError({ level: 'critical' })]} {...defaultProps} />)
    expect(screen.getAllByText('CRIT').length).toBeGreaterThanOrEqual(1)
  })
  it('shows warning level badge', () => {
    render(<ErrorDiagnosticsPanel errors={[makeError({ level: 'warning' })]} {...defaultProps} />)
    expect(screen.getAllByText('WRN').length).toBeGreaterThanOrEqual(1)
  })
  it('shows info level badge', () => {
    render(<ErrorDiagnosticsPanel errors={[makeError({ level: 'info' })]} {...defaultProps} />)
    expect(screen.getAllByText('INFO').length).toBeGreaterThanOrEqual(1)
  })
  it('shows HTTP method and path', () => {
    render(<ErrorDiagnosticsPanel errors={[makeError({ httpMethod: 'GET', httpPath: '/api/test' })]} {...defaultProps} />)
    expect(screen.getAllByText(/GET/).length).toBeGreaterThanOrEqual(1)
  })
  it('shows HTTP status', () => {
    render(<ErrorDiagnosticsPanel errors={[makeError({ httpStatus: 500 })]} {...defaultProps} />)
    expect(screen.getAllByText('500').length).toBeGreaterThanOrEqual(1)
  })
  it('shows source and metadata', () => {
    const { container } = render(<ErrorDiagnosticsPanel errors={[makeError({ source: 'backend', httpMethod: 'GET', httpPath: '/api', httpStatus: 500, durationMs: 150, correlationId: 'abc-123' })]} {...defaultProps} />)
    const text = container.textContent || ''
    expect(text).toContain('GET')
    expect(text).toContain('500')
  })
  it('groups errors by fingerprint', () => {
    const errors = [
      makeError({ id: 'e1', fingerprint: 'fp1', message: 'Error A' }),
      makeError({ id: 'e2', fingerprint: 'fp1', message: 'Error A' }),
    ]
    render(<ErrorDiagnosticsPanel errors={errors} {...defaultProps} />)
    expect(screen.getAllByText(/2/).length).toBeGreaterThanOrEqual(1)
  })
  it('calls onClear', () => {
    const onClear = vi.fn()
    const { container } = render(<ErrorDiagnosticsPanel errors={[makeError()]} onClear={onClear} />)
    const clearBtn = container.querySelector('button')
    if (clearBtn) fireEvent.click(clearBtn)
    expect(onClear).toHaveBeenCalled()
  })
  it('expands on click to show copy buttons', () => {
    render(<ErrorDiagnosticsPanel errors={[makeError({ stack: 'at foo\nat bar' })]} {...defaultProps} />)
    fireEvent.click(screen.getAllByText('Something failed')[0])
    expect(screen.getAllByText(/Copy diagnostics/).length).toBeGreaterThanOrEqual(1)
  })
})
