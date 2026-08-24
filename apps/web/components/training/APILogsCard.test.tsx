// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, within, cleanup } from '@testing-library/react'
import { APILogsCard } from './APILogsCard'

vi.mock('@/lib/training-controller', () => ({
  trainingController: {
    startFromSessionsSloNet: vi.fn().mockResolvedValue(undefined),
    streamFromSessionsSloNet: vi.fn(async function* () {}),
    cancelFromSessionsSloNet: vi.fn().mockResolvedValue(undefined),
    loadCheckpoint: vi.fn().mockResolvedValue(undefined),
  },
}))

import { trainingController } from '@/lib/training-controller'

function makeToast() {
  return vi.fn()
}

function getCard() {
  return document.querySelector('[class*="rounded-lg"][class*="border"]') as HTMLElement
}

describe('APILogsCard', () => {
  let toast: ReturnType<typeof makeToast>

  beforeEach(() => {
    toast = makeToast()
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders idle state with config inputs and start button', () => {
    render(<APILogsCard addToast={toast} />)
    expect(screen.getByText('Train from API logs')).toBeInTheDocument()
    expect(screen.getByLabelText('Epochs')).toBeInTheDocument()
    expect(screen.getByLabelText('LR')).toBeInTheDocument()
    expect(screen.getByLabelText('Embed')).toBeInTheDocument()
    expect(screen.getByLabelText('Heads')).toBeInTheDocument()
    expect(screen.getByLabelText('Layers')).toBeInTheDocument()
    const card = getCard()
    expect(within(card).getByRole('button', { name: /start training/i })).toBeInTheDocument()
  })

  it('shows default config values', () => {
    render(<APILogsCard addToast={toast} />)
    expect(screen.getByLabelText('Epochs')).toHaveValue(5)
    expect(screen.getByLabelText('Embed')).toHaveValue(128)
    expect(screen.getByLabelText('Heads')).toHaveValue(4)
    expect(screen.getByLabelText('Layers')).toHaveValue(4)
  })

  it('updates config via inputs', () => {
    render(<APILogsCard addToast={toast} />)
    const epochsInput = screen.getByLabelText('Epochs')
    fireEvent.change(epochsInput, { target: { value: '10' } })
    expect(epochsInput).toHaveValue(10)
  })

  it('calls startFromSessionsSloNet on start click', async () => {
    render(<APILogsCard addToast={toast} />)
    const card = getCard()
    fireEvent.click(within(card).getByRole('button', { name: /start training/i }))
    await vi.waitFor(() => {
      expect(trainingController.startFromSessionsSloNet).toHaveBeenCalledTimes(1)
    })
    const args = vi.mocked(trainingController.startFromSessionsSloNet).mock.calls[0][0]
    expect(args).toHaveProperty('epochs', 5)
    expect(args).toHaveProperty('learning_rate', 3e-4)
    expect(args).toHaveProperty('soul_name', 'api-logs-trained')
  })

  it('shows error state on catch', async () => {
    vi.mocked(trainingController.startFromSessionsSloNet).mockRejectedValueOnce(new Error('boom'))
    render(<APILogsCard addToast={toast} />)
    const card = getCard()
    fireEvent.click(within(card).getByRole('button', { name: /start training/i }))
    await vi.waitFor(() => {
      expect(screen.getByText('boom')).toBeInTheDocument()
    })
    expect(toast).toHaveBeenCalledWith('Could not training', 'error')
  })

  it('dismiss error returns to idle', async () => {
    vi.mocked(trainingController.startFromSessionsSloNet).mockRejectedValueOnce(new Error('x'))
    render(<APILogsCard addToast={toast} />)
    const card = getCard()
    fireEvent.click(within(card).getByRole('button', { name: /start training/i }))
    await vi.waitFor(() => screen.getByText('Training failed'))
    fireEvent.click(screen.getByRole('button', { name: /dismiss/i }))
    expect(screen.getByText('Train from API logs')).toBeInTheDocument()
  })

  it('cancel stops training and returns to idle', async () => {
    let resolveStream: () => void
    const streamPromise = new Promise<void>(r => { resolveStream = r })
    vi.mocked(trainingController.streamFromSessionsSloNet).mockImplementation(
      async function* () {
        yield { phase: 'TRAIN', status: 'working', data: { loss: 0.5 }, meta: {}, message: '' }
        await streamPromise
      } as any
    )

    render(<APILogsCard addToast={toast} />)
    const card = getCard()
    fireEvent.click(within(card).getByRole('button', { name: /start training/i }))

    await vi.waitFor(() => {
      expect(within(card).getByRole('button', { name: /stop/i })).toBeInTheDocument()
    })

    fireEvent.click(within(card).getByRole('button', { name: /stop/i }))
    expect(trainingController.cancelFromSessionsSloNet).toHaveBeenCalled()
    resolveStream!()
  })
})
