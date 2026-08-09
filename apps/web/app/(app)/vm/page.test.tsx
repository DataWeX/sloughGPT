import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within, cleanup } from '@testing-library/react'

vi.mock('@/lib/vm-controller', () => ({
  vmController: {
    run: vi.fn(),
    builtins: vi.fn(),
    info: vi.fn(),
    trainingJob: vi.fn(),
    stopTrainingJob: vi.fn(),
  },
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    list: vi.fn(),
  },
}))

import { vmController } from '@/lib/vm-controller'
import { datasetController } from '@/lib/dataset-controller'
import VMPage from './page'

const mockedRun = vi.mocked(vmController.run)
const mockedTrainingJob = vi.mocked(vmController.trainingJob)
const mockedDatasets = vi.mocked(datasetController.list)

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
    mockedDatasets.mockResolvedValue([])
    localStorage.clear()
  })

  afterEach(() => {
    cleanup()
  })

  const hintText = (text: string) => (content: string, el: Element | null) =>
    el instanceof HTMLElement &&
    el.tagName === 'LI' &&
    (el.textContent?.replace(/\s+/g, ' ').trim() ?? '') === text

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

  it('renders train program selector button', () => {
    render(<VMPage />)
    const buttons = screen.getAllByRole('button')
    const names = buttons.map((b) => b.textContent?.trim().toLowerCase())
    expect(names).toContain('train')
  })

  it('train program loads SYS_TRAIN_START source', () => {
    const { container } = render(<VMPage />)
    const trainBtns = screen.getAllByRole('button', { name: 'train' })
    fireEvent.click(trainBtns[trainBtns.length - 1])
    const textarea = container.querySelector('textarea')
    expect(textarea!.value).toContain('SYS_TRAIN_START')
    expect(textarea!.value).toContain('{"dataset":"shakespeare","epochs":1}')
  })

  it('renders train-status program selector button', () => {
    render(<VMPage />)
    const buttons = screen.getAllByRole('button')
    const names = buttons.map((b) => b.textContent?.trim().toLowerCase())
    expect(names).toContain('train-status')
  })

  it('train-status program loads SYS_TRAIN_STATUS + GET_RESULT source', () => {
    const { container } = render(<VMPage />)
    const btns = screen.getAllByRole('button', { name: 'train-status' })
    fireEvent.click(btns[btns.length - 1])
    const textarea = container.querySelector('textarea')
    expect(textarea!.value).toContain('SYS_TRAIN_STATUS')
    expect(textarea!.value).toContain('SYS_TRAIN_GET_RESULT')
    expect(textarea!.value).toContain('0x90000')
  })

  it('renders role selector with user default', () => {
    const { container } = render(<VMPage />)
    const selects = within(container).getAllByLabelText('VM role') as HTMLSelectElement[]
    expect(selects[0].value).toBe('user')
  })

  it('passes selected role to run', async () => {
    const { container } = render(<VMPage />)
    within(container).getAllByLabelText('VM role')[0] as HTMLSelectElement
    fireEvent.change(within(container).getAllByLabelText('VM role')[0], { target: { value: 'admin' } })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('halted')).toBeInTheDocument()
    })
    expect(mockedRun).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ role: 'admin' }),
    )
  })

  it('defaults run role to user', async () => {
    const { container } = render(<VMPage />)
    fireEvent.click(within(container).getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('halted')).toBeInTheDocument()
    })
    expect(mockedRun).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ role: 'user' }),
    )
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

  it('reference panel documents training syscalls', () => {
    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Ref' })[0])
    expect(screen.getByText(/EAX=28 train start/)).toBeInTheDocument()
    expect(screen.getByText(/EAX=29 train status/)).toBeInTheDocument()
    expect(screen.getByText(/EAX=30 train result/)).toBeInTheDocument()
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

  it('does not render training card without a training job', async () => {
    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('halted')).toBeInTheDocument()
    })
    expect(screen.queryByText('Training')).not.toBeInTheDocument()
    expect(mockedTrainingJob).not.toHaveBeenCalled()
  })

  it('shows running training card state', async () => {
    mockedRun.mockResolvedValue(fakeResult({ training_job_id: 7 }))
    mockedTrainingJob.mockResolvedValue({
      job_id: 7,
      api_job_id: 'abc-123',
      status: 'running',
      progress: 0.42,
    })

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('Training')).toBeInTheDocument()
    })
    expect(screen.getByText('running')).toBeInTheDocument()
    expect(screen.getByText('abc-123')).toBeInTheDocument()
    expect(screen.getByText(/Training in progress/)).toBeInTheDocument()
  })

  it('shows completed training card state and stops polling', async () => {
    mockedRun.mockResolvedValue(fakeResult({ training_job_id: 3 }))
    mockedTrainingJob.mockResolvedValue({
      job_id: 3,
      api_job_id: 'done-9',
      status: 'completed',
      progress: 1,
    })

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('Training completed successfully.')).toBeInTheDocument()
    })
    expect(screen.getByText('completed')).toBeInTheDocument()
    expect(screen.getByText('#3')).toBeInTheDocument()
  })

  it('renders the final result JSON inside the completed training card', async () => {
    mockedRun.mockResolvedValue(fakeResult({ training_job_id: 3 }))
    mockedTrainingJob.mockResolvedValue({
      job_id: 3,
      api_job_id: 'done-9',
      status: 'completed',
      progress: 1,
      result: '{"success": true, "final_loss": 1.2}',
    })

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('Training completed successfully.')).toBeInTheDocument()
    })
    expect(screen.getByText(/final_loss.*1\.2/)).toBeInTheDocument()
  })

  it('shows a Stop button on a running job and stops on click', async () => {
    mockedRun.mockResolvedValue(fakeResult({ training_job_id: 7 }))
    mockedTrainingJob.mockResolvedValue({
      job_id: 7,
      api_job_id: 'abc-123',
      status: 'running',
      progress: 0.42,
    })
    const mockedStop = vi.mocked(vmController.stopTrainingJob)
    mockedStop.mockResolvedValue({ status: 'stopping', job_id: 7 })

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('running')).toBeInTheDocument()
    })
    const stopBtn = screen.getByRole('button', { name: 'Stop' })
    fireEvent.click(stopBtn)

    await waitFor(() => {
      expect(mockedStop).toHaveBeenCalledWith(7)
    })
  })

  it('shows failed training card state with error', async () => {
    mockedRun.mockResolvedValue(fakeResult({ training_job_id: 5 }))
    mockedTrainingJob.mockResolvedValue({
      job_id: 5,
      api_job_id: '',
      status: 'failed',
      progress: 0.5,
      error: 'dataset not found',
    })

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getAllByText('dataset not found').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getByText('failed')).toBeInTheDocument()
  })

  it('shows permission denied hint when EAX is -2', async () => {
    mockedRun.mockResolvedValue(
      fakeResult({
        registers: [
          { name: 'EAX', value: -2, hex: '0xFFFFFFFE' },
          { name: 'ECX', value: 0, hex: '0x00000000' },
        ],
      }),
    )

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText(/A syscall was denied for the current role/)).toBeInTheDocument()
    })
    expect(screen.getByText(/require the role/)).toBeInTheDocument()
    expect(screen.getAllByText('admin').length).toBeGreaterThanOrEqual(1)
  })

  it('does not show permission denied hint on a normal run', async () => {
    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('halted')).toBeInTheDocument()
    })
    expect(screen.queryByText(/A syscall was denied for the current role/)).not.toBeInTheDocument()
  })

  it('renders the training result JSON from the run response', async () => {
    mockedRun.mockResolvedValue(
      fakeResult({ training_result: '{"success": true, "final_loss": 1.5}' }),
    )

    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('Training result')).toBeInTheDocument()
    })
    expect(screen.getByText(/final_loss.*1\.5/)).toBeInTheDocument()
  })

  it('does not render a training result card when absent', async () => {
    render(<VMPage />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(screen.getByText('halted')).toBeInTheDocument()
    })
    expect(screen.queryByText('Training result')).not.toBeInTheDocument()
  })

  it('renders the training launch card with default config', () => {
    const { container } = render(<VMPage />)
    expect(screen.getAllByText('Training launch').length).toBeGreaterThanOrEqual(1)
    expect(within(container).getAllByLabelText('Training dataset')[0]).toHaveValue(
      'shakespeare',
    )
    expect(within(container).getAllByLabelText('Training epochs')[0]).toHaveValue(1)
    expect(within(container).getAllByLabelText('Training embed size')[0]).toHaveValue(128)
  })

  it('load sample writes generated train source into the editor', () => {
    const { container } = render(<VMPage />)
    fireEvent.click(within(container).getAllByRole('button', { name: 'Load sample' })[0])
    const textarea = container.querySelector('textarea')
    expect(textarea!.value).toContain('SYS_TRAIN_START')
    expect(textarea!.value).toContain(
      '{"dataset":"shakespeare","epochs":1,"lr":0.001,"batch_size":32,"n_layer":4,"n_head":4,"embed_dim":128}',
    )
  })

  it('launch training runs the generated source with configured values', async () => {
    const { container } = render(<VMPage />)
    fireEvent.change(within(container).getAllByLabelText('Training dataset')[0], {
      target: { value: 'tinyshakespeare' },
    })
    fireEvent.change(within(container).getAllByLabelText('Training epochs')[0], {
      target: { value: '2' },
    })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Launch training' })[0])

    await waitFor(() => {
      expect(mockedRun).toHaveBeenCalledWith(
        expect.stringContaining('"dataset":"tinyshakespeare","epochs":2'),
        expect.objectContaining({ role: 'user' }),
      )
    })
    expect(screen.getAllByText('halted').length).toBeGreaterThanOrEqual(1)
  })

  it('launch training respects the selected role', async () => {
    const { container } = render(<VMPage />)
    fireEvent.change(within(container).getAllByLabelText('VM role')[0], {
      target: { value: 'admin' },
    })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Launch training' })[0])

    await waitFor(() => {
      expect(mockedRun).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ role: 'admin' }),
      )
    })
  })

  it('shows a launch confirmation when the job starts successfully', async () => {
    mockedRun.mockResolvedValue(fakeResult({
      registers: [
        { name: 'EAX', value: 7, hex: '0x00000007' },
        { name: 'ECX', value: 0, hex: '0x00000000' },
      ],
    }))
    const { container } = render(<VMPage />)
    fireEvent.change(within(container).getAllByLabelText('VM role')[0], {
      target: { value: 'admin' },
    })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Launch training' })[0])

    await waitFor(() => {
      expect(
        within(container).getByText(/Launched training job #7/),
      ).toBeTruthy()
    })
  })

  it('shows no launch confirmation when the job was denied', async () => {
    const { container } = render(<VMPage />)
    fireEvent.click(within(container).getAllByRole('button', { name: 'Launch training' })[0])

    await waitFor(() => {
      expect(mockedRun).toHaveBeenCalled()
    })
    expect(
      within(container).queryByText(/Launched training job/),
    ).toBeNull()
  })

  it('dismisses the launch confirmation', async () => {
    mockedRun.mockResolvedValue(fakeResult({
      registers: [
        { name: 'EAX', value: 2, hex: '0x00000002' },
        { name: 'ECX', value: 0, hex: '0x00000000' },
      ],
    }))
    const { container } = render(<VMPage />)
    fireEvent.change(within(container).getAllByLabelText('VM role')[0], {
      target: { value: 'admin' },
    })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Launch training' })[0])
    await waitFor(() => {
      expect(
        within(container).getByText(/Launched training job #2/),
      ).toBeTruthy()
    })

    fireEvent.click(within(container).getByRole('button', { name: 'Dismiss' }))
    expect(
      within(container).queryByText(/Launched training job/),
    ).toBeNull()
  })

  it('renders a dataset dropdown populated from the backend', async () => {
    mockedDatasets.mockResolvedValue([
      { name: 'shakespeare', source: 'local', size: 1 },
      { name: 'tinyshakespeare', source: 'local', size: 1 },
    ] as any)

    const { container } = render(<VMPage />)
    await waitFor(() => {
      expect(
        within(container).getAllByLabelText('Training dataset')[0].tagName,
      ).toBe('SELECT')
    })
    const select = within(container).getAllByLabelText('Training dataset')[0] as HTMLSelectElement
    expect(select.value).toBe('shakespeare')
    expect(
      Array.from(select.options).map((o) => o.value),
    ).toEqual(expect.arrayContaining(['shakespeare', 'tinyshakespeare', '__custom__']))
  })

  it('launching with a selected dataset uses it in the generated source', async () => {
    mockedDatasets.mockResolvedValue([
      { name: 'shakespeare', source: 'local', size: 1 },
      { name: 'tinyshakespeare', source: 'local', size: 1 },
    ] as any)

    const { container } = render(<VMPage />)
    await waitFor(() => {
      expect(
        within(container).getAllByLabelText('Training dataset')[0].tagName,
      ).toBe('SELECT')
    })
    fireEvent.change(within(container).getAllByLabelText('Training dataset')[0], {
      target: { value: 'tinyshakespeare' },
    })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Launch training' })[0])

    await waitFor(() => {
      expect(mockedRun).toHaveBeenCalledWith(
        expect.stringContaining('"dataset":"tinyshakespeare"'),
        expect.objectContaining({ role: 'user' }),
      )
    })
  })

  it('custom dataset option reveals a text input used at launch', async () => {
    mockedDatasets.mockResolvedValue([
      { name: 'shakespeare', source: 'local', size: 1 },
    ] as any)

    const { container } = render(<VMPage />)
    await waitFor(() => {
      expect(
        within(container).getAllByLabelText('Training dataset')[0].tagName,
      ).toBe('SELECT')
    })
    fireEvent.change(within(container).getAllByLabelText('Training dataset')[0], {
      target: { value: '__custom__' },
    })
    const inputs = within(container).getAllByLabelText('Training dataset')
    const customInput = inputs[inputs.length - 1] as HTMLInputElement
    expect(customInput.tagName).toBe('INPUT')
    fireEvent.change(customInput, { target: { value: 'mydata' } })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Launch training' })[0])

    await waitFor(() => {
      expect(mockedRun).toHaveBeenCalledWith(
        expect.stringContaining('"dataset":"mydata"'),
        expect.any(Object),
      )
    })
  })

  it('warns about an unknown custom dataset and lists available ones', async () => {
    mockedDatasets.mockResolvedValue([
      { name: 'shakespeare', source: 'local', size: 1 },
      { name: 'tinyshakespeare', source: 'local', size: 1 },
    ] as any)

    const { container } = render(<VMPage />)
    await waitFor(() => {
      expect(
        within(container).getAllByLabelText('Training dataset')[0].tagName,
      ).toBe('SELECT')
    })
    fireEvent.change(within(container).getAllByLabelText('Training dataset')[0], {
      target: { value: '__custom__' },
    })
    const inputs = within(container).getAllByLabelText('Training dataset')
    const customInput = inputs[inputs.length - 1] as HTMLInputElement
    fireEvent.change(customInput, { target: { value: 'nope' } })

    expect(within(container).getByText(/Unknown dataset "nope"/)).toBeTruthy()
    expect(within(container).getByText(/Available: shakespeare/)).toBeTruthy()
  })

  it('omits the unknown-dataset warning for a known custom name', async () => {
    mockedDatasets.mockResolvedValue([
      { name: 'shakespeare', source: 'local', size: 1 },
    ] as any)

    const { container } = render(<VMPage />)
    await waitFor(() => {
      expect(
        within(container).getAllByLabelText('Training dataset')[0].tagName,
      ).toBe('SELECT')
    })
    fireEvent.change(within(container).getAllByLabelText('Training dataset')[0], {
      target: { value: '__custom__' },
    })
    const inputs = within(container).getAllByLabelText('Training dataset')
    const customInput = inputs[inputs.length - 1] as HTMLInputElement
    fireEvent.change(customInput, { target: { value: 'shakespeare' } })

    expect(within(container).queryByText(/Unknown dataset/)).toBeNull()
  })

  it('launch clamps cleared numeric config to safe defaults', async () => {
    const { container } = render(<VMPage />)
    fireEvent.change(within(container).getAllByLabelText('Training epochs')[0], {
      target: { value: '' },
    })
    fireEvent.change(within(container).getAllByLabelText('Training learning rate')[0], {
      target: { value: '' },
    })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Launch training' })[0])

    await waitFor(() => {
      expect(mockedRun).toHaveBeenCalledWith(
        expect.stringContaining('"dataset":"shakespeare","epochs":1,"lr":0.001'),
        expect.any(Object),
      )
    })
  })

  it('persists the training config across reloads', () => {
    const first = render(<VMPage />)
    fireEvent.change(within(first.container).getAllByLabelText('Training epochs')[0], {
      target: { value: '5' },
    })
    fireEvent.change(within(first.container).getAllByLabelText('Training dataset')[0], {
      target: { value: 'tinyshakespeare' },
    })
    cleanup()

    const second = render(<VMPage />)
    expect(
      (within(second.container).getAllByLabelText('Training epochs')[0] as HTMLInputElement)
        .value,
    ).toBe('5')
    expect(
      (within(second.container).getAllByLabelText('Training dataset')[0] as HTMLInputElement)
        .value,
    ).toBe('tinyshakespeare')
    const saved = JSON.parse(localStorage.getItem('vm-train-config') ?? '{}')
    expect(saved.epochs).toBe(5)
    expect(saved.dataset).toBe('tinyshakespeare')
  })

  it('reset config restores defaults and clears a custom dataset', async () => {
    mockedDatasets.mockResolvedValue([
      { name: 'shakespeare', source: 'local', size: 1 },
    ] as any)

    const { container } = render(<VMPage />)
    await waitFor(() => {
      expect(
        within(container).getAllByLabelText('Training dataset')[0].tagName,
      ).toBe('SELECT')
    })
    fireEvent.change(within(container).getAllByLabelText('Training epochs')[0], {
      target: { value: '12' },
    })
    fireEvent.change(within(container).getAllByLabelText('Training dataset')[0], {
      target: { value: '__custom__' },
    })
    const customInput = within(container).getAllByLabelText('Training dataset')[
      within(container).getAllByLabelText('Training dataset').length - 1
    ] as HTMLInputElement
    fireEvent.change(customInput, { target: { value: 'mydata' } })

    fireEvent.click(within(container).getAllByRole('button', { name: 'Reset config' })[0])

    expect(
      (within(container).getAllByLabelText('Training epochs')[0] as HTMLInputElement).value,
    ).toBe('1')
    expect(
      (within(container).getAllByLabelText('Training dataset')[0] as HTMLSelectElement).value,
    ).toBe('shakespeare')
    const datasetInputs = within(container).getAllByLabelText('Training dataset')
    expect(datasetInputs.length).toBe(1)
    const saved = JSON.parse(localStorage.getItem('vm-train-config') ?? '{}')
    expect(saved.epochs).toBe(1)
    expect(saved.dataset).toBe('shakespeare')
  })

  it('warns and offers switch to admin when the role is user', () => {
    const { container } = render(<VMPage />)
    expect(
      within(container).getByText(/Training is denied for the user role/),
    ).toBeTruthy()
    fireEvent.click(within(container).getAllByRole('button', { name: 'Switch to admin' })[0])
    expect(
      (within(container).getAllByLabelText('VM role')[0] as HTMLSelectElement).value,
    ).toBe('admin')
    expect(localStorage.getItem('vm-role')).toBe('admin')
  })

  it('hides the user-role warning for admin and kernel roles', () => {
    const { container } = render(<VMPage />)
    const roleSelect = within(container).getAllByLabelText('VM role')[0]
    fireEvent.change(roleSelect, { target: { value: 'admin' } })
    expect(within(container).queryByText(/Training is denied for the user role/)).toBeNull()

    fireEvent.change(roleSelect, { target: { value: 'kernel' } })
    expect(within(container).queryByText(/Training is denied for the user role/)).toBeNull()
  })

  it('shows fallback hints when config fields are cleared or invalid', () => {
    const { container } = render(<VMPage />)
    fireEvent.change(within(container).getAllByLabelText('Training epochs')[0], {
      target: { value: '' },
    })
    fireEvent.change(within(container).getAllByLabelText('Training learning rate')[0], {
      target: { value: '' },
    })
    fireEvent.change(within(container).getAllByLabelText('Training dataset')[0], {
      target: { value: '' },
    })
    expect(within(container).getByText(hintText('Epochs: using default 1'))).toBeTruthy()
    expect(
      within(container).getByText(hintText('Learning rate: using default 0.001')),
    ).toBeTruthy()
    expect(
      within(container).getByText(hintText('Dataset: using default "shakespeare"')),
    ).toBeTruthy()
  })

  it('clears fallback hints once fields are valid again', () => {
    const { container } = render(<VMPage />)
    const epochs = within(container).getAllByLabelText('Training epochs')[0]
    fireEvent.change(epochs, { target: { value: '' } })
    expect(within(container).queryByText(hintText('Epochs: using default 1'))).not.toBeNull()

    fireEvent.change(epochs, { target: { value: '3' } })
    expect(within(container).queryByText(hintText('Epochs: using default 1'))).toBeNull()
  })

  it('clamps a cleared steps field to 1 on run', async () => {
    const { container } = render(<VMPage />)
    fireEvent.change(within(container).getByLabelText('Steps:'), {
      target: { value: '' },
    })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(mockedRun).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ maxSteps: 1 }),
      )
    })
  })

  it('clamps an oversized steps field to the limit on run', async () => {
    const { container } = render(<VMPage />)
    fireEvent.change(within(container).getByLabelText('Steps:'), {
      target: { value: '9999999999' },
    })
    fireEvent.click(within(container).getAllByRole('button', { name: 'Run' })[0])

    await waitFor(() => {
      expect(mockedRun).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ maxSteps: 1000000 }),
      )
    })
  })

  it('persists the selected role across reloads', async () => {
    const first = render(<VMPage />)
    fireEvent.change(within(first.container).getAllByLabelText('VM role')[0], {
      target: { value: 'admin' },
    })
    cleanup()

    const second = render(<VMPage />)
    const select = within(second.container).getAllByLabelText('VM role')[0] as HTMLSelectElement
    expect(select.value).toBe('admin')
    expect(localStorage.getItem('vm-role')).toBe('admin')
  })

  it('persists the steps value across reloads', async () => {
    const first = render(<VMPage />)
    fireEvent.change(within(first.container).getByLabelText('Steps:'), {
      target: { value: '250' },
    })
    cleanup()

    const second = render(<VMPage />)
    expect(
      (within(second.container).getByLabelText('Steps:') as HTMLInputElement).value,
    ).toBe('250')
    expect(localStorage.getItem('vm-max-steps')).toBe('250')
  })
})
