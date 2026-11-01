import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { CodeBlock } from './CodeBlock'

vi.mock('@/lib/constants', () => ({
  COPY_FEEDBACK_DURATION_MS: 2000,
}))

vi.mock('prismjs', () => ({
  default: {
    languages: { javascript: {} },
    highlight: (code: string) => `<span class="token">${code}</span>`,
  },
}))

beforeEach(() => {
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
    writable: true,
  })
})

afterEach(cleanup)

describe('CodeBlock', () => {
  it('renders without crashing', () => {
    render(<CodeBlock language="javascript" code="const x = 1" />)
    expect(screen.getByText('javascript')).toBeInTheDocument()
  })

  it('displays the code content', () => {
    render(<CodeBlock language="python" code="print('hello')" />)
    expect(screen.getByText("print('hello')")).toBeInTheDocument()
  })

  it('shows Copy button', () => {
    render(<CodeBlock language="js" code="test" />)
    expect(screen.getByText('Copy')).toBeInTheDocument()
  })

  it('copies code to clipboard on click', async () => {
    render(<CodeBlock language="js" code="const x = 1" />)
    const copyBtn = screen.getByRole('button', { name: /copy code/i })
    await act(async () => {
      fireEvent.click(copyBtn)
    })
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('const x = 1')
  })

  it('shows Copied feedback after copy', async () => {
    vi.useFakeTimers()
    render(<CodeBlock language="js" code="test" />)
    const copyBtn = screen.getByRole('button', { name: /copy code/i })
    await act(async () => {
      fireEvent.click(copyBtn)
    })
    expect(screen.getByText('Copied')).toBeInTheDocument()
    act(() => { vi.advanceTimersByTime(2100) })
    expect(screen.queryByText('Copied')).not.toBeInTheDocument()
    vi.useRealTimers()
  })

  it('shows "code" label when language is empty', () => {
    render(<CodeBlock language="" code="hello" />)
    expect(screen.getByText('code')).toBeInTheDocument()
  })
})
