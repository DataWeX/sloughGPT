import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ToolApprovalDialog } from './ToolApprovalDialog'

afterEach(cleanup)

describe('ToolApprovalDialog', () => {
  it('renders tool name', () => {
    render(
      <ToolApprovalDialog
        toolName="calculator"
        onApprove={vi.fn()}
      />
    )
    expect(screen.getByText('calculator')).toBeInTheDocument()
  })

  it('renders args when provided', () => {
    render(
      <ToolApprovalDialog
        toolName="calculator"
        args={{ expression: '2+2' }}
        onApprove={vi.fn()}
      />
    )
    expect(screen.getByText('expression: 2+2')).toBeInTheDocument()
  })

  it('calls onApprove(true) when approve clicked', () => {
    const onApprove = vi.fn()
    render(
      <ToolApprovalDialog
        toolName="calculator"
        onApprove={onApprove}
      />
    )
    const buttons = screen.getAllByRole('button')
    const approveButton = buttons.find(b => b.textContent === 'Approve')
    fireEvent.click(approveButton!)
    expect(onApprove).toHaveBeenCalledWith(true)
  })

  it('calls onApprove(false) when deny clicked', () => {
    const onApprove = vi.fn()
    render(
      <ToolApprovalDialog
        toolName="calculator"
        onApprove={onApprove}
      />
    )
    const buttons = screen.getAllByRole('button')
    const denyButton = buttons.find(b => b.textContent === 'Deny')
    fireEvent.click(denyButton!)
    expect(onApprove).toHaveBeenCalledWith(false)
  })

  it('hides after decision', () => {
    const onApprove = vi.fn()
    const { unmount } = render(
      <ToolApprovalDialog
        toolName="calculator"
        onApprove={onApprove}
      />
    )
    unmount()
    expect(screen.queryByText('calculator')).not.toBeInTheDocument()
  })
})
