// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/ui/dialog', () => {
  function Dialog({ children, open }: any) { return open ? <div data-testid="dialog">{children}</div> : null }
  function DialogContent({ children }: any) { return <div data-testid="content">{children}</div> }
  function DialogHeader({ children }: any) { return <div>{children}</div> }
  function DialogTitle({ children }: any) { return <div>{children}</div> }
  function DialogPortal({ children }: any) { return <>{children}</> }
  function DialogOverlay() { return <div data-testid="overlay" /> }
  return { Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay }
})

import { SystemPromptDialog } from './SystemPromptDialog'
import { Button } from '@/components/ui/button'

vi.mock('@/components/ui/button', () => ({
  Button: vi.fn(({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>{children}</button>
  )),
}))

describe('SystemPromptDialog', () => {
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

  it('calls onSave with draft value when Save clicked', () => {
    const onSave = vi.fn()
    render(<SystemPromptDialog open={true} onOpenChange={vi.fn()} value="" onSave={onSave} />)
    const textarea = screen.getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: 'Be concise' } })
    fireEvent.click(screen.getByText('Save'))
    expect(onSave).toHaveBeenCalledWith('Be concise')
  })

  it('closes dialog when Save clicked', () => {
    const onOpenChange = vi.fn()
    render(<SystemPromptDialog open={true} onOpenChange={onOpenChange} value="" onSave={vi.fn()} />)
    fireEvent.click(screen.getByText('Save'))
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
