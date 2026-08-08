import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

vi.mock('@/lib/vm-controller', () => ({
  vmController: {
    run: vi.fn(),
    builtins: vi.fn(),
    info: vi.fn(),
  },
}))

import { vmController } from '@/lib/vm-controller'
import VMPage from './page'

const mockedRun = vi.mocked(vmController.run)

function fakeResult(overrides: Record<string, any> = {}) {
  return {
    success: true,
    exit_code: 0,
    steps_executed: 10,
    elapsed_ms: 1.2,
    output: '',
    registers: [
      { name: 'EAX', value: 0, hex: '0x00000000' },
      { name: 'ECX', value: 0, hex: '0x00000000' },
    ],
    eip: 0,
    eip_hex: '0x00000000',
    status: 'halted',
    ...overrides,
  }
}

describe('VMPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockedRun.mockResolvedValue(fakeResult())
    localStorage.clear()
  })

  it('renders page header', () => {
    render(<VMPage />)
    expect(screen.getAllByText('VM Console').length).toBeGreaterThanOrEqual(1)
  })

  it('renders default source code in textarea', () => {
    const { container } = render(<VMPage />)
    const textarea = container.querySelector('textarea')
    expect(textarea).toBeInTheDocument()
    expect(textarea!.value).toContain('Hello, VM!')
  })

  it('renders program selector buttons', () => {
    render(<VMPage />)
    const buttons = screen.getAllByRole('button')
    const names = buttons.map((b) => b.textContent?.trim().toLowerCase())
    expect(names).toContain('hello')
    expect(names).toContain('count')
    expect(names).toContain('fib')
    expect(names).toContain('sort')
  })

  it('switches program on button click', async () => {
    const { container } = render(<VMPage />)
    const countBtns = screen.getAllByRole('button', { name: 'count' })
    await fireEvent.click(countBtns[countBtns.length - 1])
    const textarea = container.querySelector('textarea')
    expect(textarea!.value).toContain('; Count 0-9')
  })

  it('runs assembly and shows result', async () => {
    mockedRun.mockResolvedValue(fakeResult({
      registers: [{ name: 'EAX', value: 42, hex: '0x0000002A' }],
    }))

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('halted')).toBeInTheDocument()
    })
    expect(mockedRun).toHaveBeenCalled()
  })

  it('shows error banner on failure', async () => {
    mockedRun.mockResolvedValue(fakeResult({
      success: false,
      status: 'error',
      error: 'assembly failed',
    }))

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getAllByText('assembly failed').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows step limit warning', async () => {
    mockedRun.mockResolvedValue(fakeResult({ steps_executed: 5000 }))

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText(/Step limit reached/)).toBeInTheDocument()
    })
  })

  it('shows registers card after run', async () => {
    mockedRun.mockResolvedValue(fakeResult({
      registers: [
        { name: 'EAX', value: 1, hex: '0x00000001' },
        { name: 'ESP', value: 0xBFF00, hex: '0x000BFF00' },
      ],
      eip: 10,
      eip_hex: '0x0000000A',
    }))

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('Registers')).toBeInTheDocument()
    })
    expect(screen.getByText('EAX')).toBeInTheDocument()
    expect(screen.getByText('0x00000001')).toBeInTheDocument()
    expect(screen.getByText('EIP')).toBeInTheDocument()
  })

  it('shows VGA display after run', async () => {
    mockedRun.mockResolvedValue(fakeResult({ vga_text: 'Hello, VM!' }))

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('Screen (VGA 0xB8000)')).toBeInTheDocument()
    })
  })

  it('shows trace when debug enabled', async () => {
    mockedRun.mockResolvedValue(fakeResult({
      trace: [
        { step: 0, eip: '0x00000000', opcode: 'MOV', operands: 'EAX, 42' },
        { step: 1, eip: '0x00000002', opcode: 'HLT', operands: '' },
      ],
    }))

    render(<VMPage />)
    const checkboxes = screen.getAllByRole('checkbox')
    fireEvent.click(checkboxes[0])
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText(/Execution Trace/)).toBeInTheDocument()
    })
    expect(screen.getByText('MOV')).toBeInTheDocument()
    expect(screen.getByText('HLT')).toBeInTheDocument()
  })

  it('clears result on Clear button', async () => {
    mockedRun.mockResolvedValue(fakeResult())

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('halted')).toBeInTheDocument()
    })

    fireEvent.click(screen.getAllByRole('button', { name: 'Clear' })[0])
    expect(screen.queryByText('halted')).not.toBeInTheDocument()
  })

  it('toggles reference panel', () => {
    render(<VMPage />)
    expect(screen.queryByText('x86 Reference')).not.toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: 'Ref' })[0])
    expect(screen.getByText('x86 Reference')).toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: 'Ref' })[0])
    expect(screen.queryByText('x86 Reference')).not.toBeInTheDocument()
  })

  it('disables Run button while running', async () => {
    let resolveRun: any
    mockedRun.mockImplementation(() => new Promise((r) => { resolveRun = r }))

    render(<VMPage />)
    const runBtn = screen.getAllByRole('button', { name: 'Run' })[0]
    fireEvent.click(runBtn)

    await waitFor(() => {
      expect(runBtn).toBeDisabled()
    })
    resolveRun(fakeResult())
  })

  it('saves and loads source from localStorage', () => {
    localStorage.setItem('vm-source', 'SAVED CODE')
    const { container } = render(<VMPage />)
    const textarea = container.querySelector('textarea')
    expect(textarea!.value).toBe('SAVED CODE')
  })

  it('renders output card when output present', async () => {
    mockedRun.mockResolvedValue(fakeResult({ output: 'Hello, World!\n' }))

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('Output')).toBeInTheDocument()
    })
    expect(screen.getByText('Hello, World!')).toBeInTheDocument()
  })

  it('renders memory dump in debug mode', async () => {
    mockedRun.mockResolvedValue(fakeResult({
      memory_dump: '000BFF00  48 65 6C 6C 6F  Hello',
    }))

    render(<VMPage />)
    const checkboxes = screen.getAllByRole('checkbox')
    fireEvent.click(checkboxes[0])
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText(/Memory \(stack area\)/)).toBeInTheDocument()
    })
  })
})
