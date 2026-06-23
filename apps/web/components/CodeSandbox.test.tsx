// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockExecuteCode } = vi.hoisted(() => ({
  mockExecuteCode: vi.fn(),
}))

vi.mock('@/lib/piston-api', () => ({
  executeCode: mockExecuteCode,
}))

import { CodeSandbox } from './CodeSandbox'

describe('CodeSandbox', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders code in pre', () => {
    render(<CodeSandbox code="print('hi')" language="python" />)
    expect(screen.getByText("print('hi')")).toBeDefined()
  })

  it('shows language label', () => {
    render(<CodeSandbox code="print('hi')" language="python" />)
    expect(screen.getByText('python')).toBeDefined()
  })

  it('shows initial empty state', () => {
    render(<CodeSandbox code="print('hi')" />)
    expect(screen.getByText('Click Run to execute...')).toBeDefined()
  })

  it('executes JavaScript and shows output', async () => {
    const code = 'console.log("hello from js")'
    render(<CodeSandbox code={code} language="javascript" />)
    fireEvent.click(screen.getByRole('button', { name: 'Run code' }))
    await waitFor(() => {
      expect(screen.getByText('hello from js')).toBeDefined()
    })
  })

  it('shows JavaScript error output', async () => {
    const code = 'throw new Error("boom")'
    render(<CodeSandbox code={code} language="js" />)
    fireEvent.click(screen.getByRole('button', { name: 'Run code' }))
    await waitFor(() => {
      expect(screen.getByText('boom')).toBeDefined()
    })
  })

  it('executes Python via piston-api', async () => {
    mockExecuteCode.mockResolvedValue({ output: 'hello from py', error: undefined })
    render(<CodeSandbox code="print('hello')" language="python" />)
    fireEvent.click(screen.getByRole('button', { name: 'Run code' }))
    await waitFor(() => {
      expect(screen.getByText('hello from py')).toBeDefined()
    })
  })

  it('shows Python error from piston-api', async () => {
    mockExecuteCode.mockResolvedValue({ output: '', error: 'SyntaxError' })
    render(<CodeSandbox code="print(" language="python" />)
    fireEvent.click(screen.getByRole('button', { name: 'Run code' }))
    await waitFor(() => {
      expect(screen.getByText('SyntaxError')).toBeDefined()
    })
  })

  it('shows unsupported language message', () => {
    render(<CodeSandbox code="code" language="ruby" />)
    fireEvent.click(screen.getByRole('button', { name: 'Run code' }))
    expect(screen.getByText(/not supported/)).toBeDefined()
  })

  it('shows Close button when onClose is provided', () => {
    render(<CodeSandbox code="code" onClose={vi.fn()} />)
    expect(screen.getByText('Close')).toBeDefined()
  })

  it('calls onClose on Close click', () => {
    const onClose = vi.fn()
    render(<CodeSandbox code="code" onClose={onClose} />)
    fireEvent.click(screen.getByText('Close'))
    expect(onClose).toHaveBeenCalled()
  })
})
