// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value },
    removeItem: (key: string) => { delete store[key] },
    clear: () => { store = {} },
  }
})()

vi.stubGlobal('localStorage', localStorageMock)

vi.mock('@/components/ui/dialog', () => {
  function Dialog({ children, open }: any) { return open ? <div data-testid="dialog">{children}</div> : null }
  function DialogContent({ children }: any) { return <div data-testid="content">{children}</div> }
  function DialogHeader({ children }: any) { return <div>{children}</div> }
  function DialogTitle({ children }: any) { return <div>{children}</div> }
  function DialogPortal({ children }: any) { return <>{children}</> }
  function DialogOverlay() { return <div data-testid="overlay" /> }
  return { Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay }
})

vi.mock('@/components/ui/input', () => ({
  Input: ({ value, onChange, onKeyDown, ...props }: any) => (
    <input value={value} onChange={onChange} onKeyDown={onKeyDown} {...props} />
  ),
}))

import { SystemPromptDialog } from './SystemPromptDialog'

vi.mock('@/components/ui/button', () => ({
  Button: vi.fn(({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>{children}</button>
  )),
}))

describe('SystemPromptDialog', () => {
  beforeEach(() => localStorageMock.clear())
  afterEach(cleanup)

  it('renders when open', () => {
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={vi.fn()} />)
    expect(screen.getByTestId('dialog')).toBeDefined()
    expect(screen.getByText('Custom System Prompt')).toBeDefined()
  })

  it('does not render when closed', () => {
    render(<SystemPromptDialog open={false} onOpenChange={vi.fn()} value="" onSave={vi.fn()} />)
    expect(screen.queryByTestId('dialog')).toBeNull()
  })

  it('shows the current value in textarea', () => {
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="Be helpful" onSave={vi.fn()} />)
    const textarea = screen.getByLabelText('System prompt') as HTMLTextAreaElement
    expect(textarea.value).toBe('Be helpful')
  })

  it('shows preset buttons', () => {
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={vi.fn()} />)
    expect(screen.getByText('Helpful Assistant')).toBeDefined()
    expect(screen.getByText('Code Expert')).toBeDefined()
  })

  it('applies preset text on preset click', () => {
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={vi.fn()} />)
    fireEvent.click(screen.getByText('Helpful Assistant'))
    const textarea = screen.getByLabelText('System prompt') as HTMLTextAreaElement
    expect(textarea.value).toContain('helpful')
  })

  it('calls onSave with draft value when Save changes clicked', () => {
    const onSave = vi.fn()
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={onSave} />)
    const textarea = screen.getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: 'Be concise' } })
    fireEvent.click(screen.getByText('Save changes'))
    expect(onSave).toHaveBeenCalledWith('Be concise')
  })

  it('closes dialog when Save clicked', () => {
    const onOpenChange = vi.fn()
    render(<SystemPromptDialog open={true} onOpenChange={onOpenChange} value="" onSave={vi.fn()} />)
    const textarea = screen.getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: 'Be concise' } })
    fireEvent.click(screen.getByText('Save changes'))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('does not save draft when Cancel clicked', () => {
    const onSave = vi.fn()
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={onSave} />)
    const textarea = screen.getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: 'Be concise' } })
    fireEvent.click(screen.getByText('Cancel'))
    expect(onSave).not.toHaveBeenCalled()
  })
})
