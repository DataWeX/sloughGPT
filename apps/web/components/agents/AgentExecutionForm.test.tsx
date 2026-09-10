// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { AgentExecutionForm } from './AgentExecutionForm'

afterEach(() => cleanup())

const defaultProps = {
  agentName: 'Researcher',
  prompt: '',
  result: null,
  toolsUsed: [],
  running: false,
  errors: {},
  onPromptChange: vi.fn(),
  onExecute: vi.fn(),
  onClose: vi.fn(),
}

describe('AgentExecutionForm', () => {
  it('renders the form container', () => {
    render(<AgentExecutionForm {...defaultProps} />)
    expect(screen.getByTestId('agent-execution-form')).toBeTruthy()
  })

  it('renders Execute and Close buttons', () => {
    render(<AgentExecutionForm {...defaultProps} />)
    expect(screen.getByText('Execute')).toBeTruthy()
    expect(screen.getByText('Close')).toBeTruthy()
  })

  it('calls onExecute when Execute clicked', () => {
    const onExecute = vi.fn()
    render(<AgentExecutionForm {...defaultProps} prompt="hello" onExecute={onExecute} />)
    fireEvent.click(screen.getByText('Execute'))
    expect(onExecute).toHaveBeenCalled()
  })

  it('calls onClose when Close clicked', () => {
    const onClose = vi.fn()
    render(<AgentExecutionForm {...defaultProps} onClose={onClose} />)
    fireEvent.click(screen.getByText('Close'))
    expect(onClose).toHaveBeenCalled()
  })

  it('disables Execute when prompt is empty', () => {
    render(<AgentExecutionForm {...defaultProps} prompt="" />)
    expect(screen.getByText('Execute').hasAttribute('disabled')).toBe(true)
  })

  it('shows Running... when executing', () => {
    render(<AgentExecutionForm {...defaultProps} running={true} prompt="hi" />)
    expect(screen.getByText('Running...')).toBeTruthy()
    expect(screen.getByText('Running...').hasAttribute('disabled')).toBe(true)
  })

  it('displays result when provided', () => {
    render(<AgentExecutionForm {...defaultProps} result="Done!" />)
    expect(screen.getByText('Response')).toBeTruthy()
    expect(screen.getByText('Done!')).toBeTruthy()
  })

  it('displays tools used', () => {
    render(
      <AgentExecutionForm
        {...defaultProps}
        toolsUsed={[{ tool: 'web_search', result: null }]}
        result="Ok"
      />,
    )
    expect(screen.getByText('web search')).toBeTruthy()
  })

  it('shows validation error', () => {
    render(<AgentExecutionForm {...defaultProps} errors={{ prompt: 'Prompt required' }} />)
    expect(screen.getByText('Prompt required')).toBeTruthy()
    expect(screen.getByRole('alert')).toBeTruthy()
  })
})
