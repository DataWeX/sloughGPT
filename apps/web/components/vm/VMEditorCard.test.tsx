/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { VMEditorCard } from './VMEditorCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  Textarea: (props: any) => <textarea data-testid="source-editor" {...props} />,
}))

describe('VMEditorCard', () => {
  const defaultProps = {
    source: '[BITS 32]\nMOV EAX, 42\nHLT',
    onSourceChange: vi.fn(),
    onRun: vi.fn(),
  }

  it('renders the title', () => {
    render(<VMEditorCard {...defaultProps} />)
    expect(screen.getByText('Assembly Source')).toBeDefined()
  })

  it('renders the source editor with content', () => {
    render(<VMEditorCard {...defaultProps} />)
    expect((screen.getByTestId('source-editor') as HTMLTextAreaElement).value).toContain('MOV EAX, 42')
  })

  it('renders Run button', () => {
    render(<VMEditorCard {...defaultProps} />)
    expect(screen.getByText('Run')).toBeDefined()
  })

  it('calls onRun when Run is clicked', () => {
    const onRun = vi.fn()
    render(<VMEditorCard {...defaultProps} onRun={onRun} />)
    fireEvent.click(screen.getByText('Run'))
    expect(onRun).toHaveBeenCalledOnce()
  })

  it('calls onSourceChange when editor changes', () => {
    const onSourceChange = vi.fn()
    render(<VMEditorCard {...defaultProps} onSourceChange={onSourceChange} />)
    fireEvent.change(screen.getByTestId('source-editor'), { target: { value: 'NEW' } })
    expect(onSourceChange).toHaveBeenCalledWith('NEW')
  })

  it('disables Run button when running', () => {
    render(<VMEditorCard {...defaultProps} running />)
    const btn = screen.getByText('Running...')
    expect(btn).toBeDefined()
  })

  it('renders program buttons when programs provided', () => {
    const programs = { hello: 'code1', count: 'code2' }
    render(<VMEditorCard {...defaultProps} programs={programs} onProgramSelect={vi.fn()} />)
    expect(screen.getByText('hello')).toBeDefined()
    expect(screen.getByText('count')).toBeDefined()
  })

  it('renders Step button when onStep provided', () => {
    render(<VMEditorCard {...defaultProps} onStep={vi.fn()} />)
    expect(screen.getByText('Step')).toBeDefined()
  })
})
