import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { CodeExecutionResults } from './CodeExecutionResults'

afterEach(cleanup)
beforeEach(() => {
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      writeText: vi.fn().mockResolvedValue(undefined),
    },
    writable: true,
  })
})

const mockExecutions = [
  {
    id: '1',
    code: 'print("Hello")',
    language: 'python',
    output: 'Hello',
    exitCode: 0,
    duration: 150,
    timestamp: new Date(),
  },
  {
    id: '2',
    code: 'console.log("World")',
    language: 'javascript',
    output: 'World',
    exitCode: 0,
    duration: 50,
    timestamp: new Date(),
  },
  {
    id: '3',
    code: 'invalid code',
    language: 'python',
    output: '',
    error: 'SyntaxError: invalid syntax',
    exitCode: 1,
    duration: 10,
    timestamp: new Date(),
  },
]

describe('CodeExecutionResults', () => {
  it('renders empty state', () => {
    render(<CodeExecutionResults executions={[]} />)
    expect(screen.getByText('No code executions')).toBeInTheDocument()
  })

  it('renders execution count', () => {
    render(<CodeExecutionResults executions={mockExecutions} />)
    const pythons = screen.getAllByText('python')
    expect(pythons.length).toBe(2)
    expect(screen.getByText('javascript')).toBeInTheDocument()
  })

  it('shows success icon for exit code 0', () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    expect(screen.getByText('✓')).toBeInTheDocument()
  })

  it('shows error icon for non-zero exit code', () => {
    render(<CodeExecutionResults executions={[mockExecutions[2]]} />)
    expect(screen.getByText('✗')).toBeInTheDocument()
  })

  it('expands execution on click', () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    fireEvent.click(screen.getByText('python'))
    expect(screen.getByText('print("Hello")')).toBeInTheDocument()
    expect(screen.getByText('Output')).toBeInTheDocument()
  })

  it('shows error panel for failed execution', () => {
    render(<CodeExecutionResults executions={[mockExecutions[2]]} />)
    fireEvent.click(screen.getByText('python'))
    expect(screen.getByText('Error')).toBeInTheDocument()
    expect(screen.getByText('SyntaxError: invalid syntax')).toBeInTheDocument()
  })

  it('calls onRerun when rerun clicked', () => {
    const onRerun = vi.fn()
    render(<CodeExecutionResults executions={[mockExecutions[0]]} onRerun={onRerun} />)
    fireEvent.click(screen.getByText('python'))
    fireEvent.click(screen.getByText('Rerun'))
    expect(onRerun).toHaveBeenCalledWith('1')
  })

  it('copies code to clipboard', async () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    fireEvent.click(screen.getByText('python'))
    const copyBtn = screen.getAllByRole('button')[0]
    fireEvent.click(copyBtn)
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('print("Hello")')
  })

  it('collapses expanded execution', () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    fireEvent.click(screen.getByText('python'))
    expect(screen.getByText('print("Hello")')).toBeInTheDocument()
    fireEvent.click(screen.getByText('python'))
    expect(screen.queryByText('print("Hello")')).not.toBeInTheDocument()
  })

  it('shows duration', () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    expect(screen.getByText('150ms')).toBeInTheDocument()
  })
})