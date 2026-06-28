// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ToolCallPanel } from './ToolCallPanel'
import type { ToolCallEvent } from '@/lib/stream-chat-response'

afterEach(cleanup)

describe('ToolCallPanel', () => {
  it('renders nothing when events are empty', () => {
    const { container } = render(<ToolCallPanel events={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders a tool call event', () => {
    render(<ToolCallPanel events={[{ tool: 'calculator', status: 'success', output: '4' }]} />)
    expect(screen.getByText('calculator')).toBeTruthy()
    expect(screen.getByText(/Done/)).toBeTruthy()
  })

  it('shows running status for executing tool', () => {
    render(<ToolCallPanel events={[{ tool: 'web_search', status: 'executing' }]} />)
    expect(screen.getByText('web_search')).toBeTruthy()
    expect(screen.getByText('Running...')).toBeTruthy()
  })

  it('shows error status for failed tool', () => {
    render(<ToolCallPanel events={[{ tool: 'run_code', status: 'error', error: 'SyntaxError' }]} />)
    expect(screen.getByText('run_code')).toBeTruthy()
    expect(screen.getByText('Failed')).toBeTruthy()
  })

  it('shows output when expanded', () => {
    render(<ToolCallPanel events={[{ tool: 'calculator', status: 'success', output: '42', duration_ms: 50 }]} />)
    expect(screen.getByText(/\(0\.1s\)/)).toBeTruthy()
  })

  it('renders multiple events in order', () => {
    const events: ToolCallEvent[] = [
      { tool: 'web_search', status: 'success', output: 'results' },
      { tool: 'calculator', status: 'success', output: '4' },
    ]
    const { container } = render(<ToolCallPanel events={events} />)
    const cards = container.querySelectorAll('button')
    expect(cards.length).toBe(2)
    expect(cards[0].textContent).toContain('web_search')
    expect(cards[1].textContent).toContain('calculator')
  })

  it('shows error details when expanded', () => {
    render(<ToolCallPanel events={[{ tool: 'run_code', status: 'error', error: 'SyntaxError: invalid syntax' }]} />)
    const buttons = screen.getAllByText('run_code')
    const button = buttons[0].closest('button')!
    fireEvent.click(button)
    expect(screen.getByText('SyntaxError: invalid syntax')).toBeTruthy()
  })

  it('shows output details when expanded', () => {
    render(<ToolCallPanel events={[{ tool: 'calculator', status: 'success', output: 'The answer is 42' }]} />)
    const buttons = screen.getAllByText('calculator')
    const button = buttons[0].closest('button')!
    fireEvent.click(button)
    expect(screen.getByText('The answer is 42')).toBeTruthy()
  })
})
