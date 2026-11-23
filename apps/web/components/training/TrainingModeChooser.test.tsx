// @vitest-environment jsdom
/**
 * Mode chooser guard — card 20261119_130319
 * ("training page: 'How do you want to train?' mode chooser").
 *
 * The chooser is an inline widget on the existing /training journey page —
 * it exists because the page only surfaced datasets. Two invariants:
 *
 *  1. Every card does something REAL. A card is either a hand-off into a
 *     flow that already exists (guided setup / fast-train fold) or it runs
 *     the from-chats training that already has an HTTP entry. DPO and
 *     chat-train stay off the chooser until they have one — a dead button
 *     is worse than a missing one.
 *  2. Plain language (UX_FLOWS Feature #12 "What they should NOT see" +
 *     the translation table): no "Epoch", no "LoRA", no learning rate /
 *     batch size / hyperparameters. Assertions are scoped to the widget's
 *     own root — never a whole-page scan.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockCtl = vi.hoisted(() => ({
  startFromSessionsSloNet: vi.fn(),
  streamFromSessionsSloNet: vi.fn(),
  cancelFromSessionsSloNet: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: mockCtl,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.filter(Boolean).join(' '),
    Card: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    CardDescription: ({ children }: any) => <p>{children}</p>,
    CardContent: passthrough,
    Button: ({ children, onClick, disabled, className }: any) => (
      <button type="button" onClick={onClick} disabled={disabled} className={className}>
        {children}
      </button>
    ),
    Progress: ({ value, label, indeterminate }: any) => (
      <div
        role="progressbar"
        aria-label={label}
        data-value={String(value)}
        data-indeterminate={String(!!indeterminate)}
      />
    ),
  }
})

import { TrainingModeChooser } from './TrainingModeChooser'

type SseEvent = {
  status: string
  message?: string
  data?: Record<string, unknown>
}

async function* sse(events: SseEvent[]) {
  for (const e of events) yield e
}

function renderChooser(overrides: Partial<React.ComponentProps<typeof TrainingModeChooser>> = {}) {
  const props = {
    inputMode: 'dataset' as const,
    onPickDataset: vi.fn(),
    onPickText: vi.fn(),
    onPickQuick: vi.fn(),
    ...overrides,
  }
  const utils = render(<TrainingModeChooser {...props} />)
  return { ...utils, props }
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
})

describe('TrainingModeChooser', () => {
  it('offers four plain-language ways to train', () => {
    const { container } = renderChooser()

    expect(screen.getByRole('button', { name: /Train on a dataset/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Paste text/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Quick train/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Train from your chats/ })).toBeTruthy()

    // Methods are folded into the descriptions, and the native-vs-external
    // model axis is named — the chooser replaces the jargon, it doesn't
    // re-label it.
    const text = container.textContent ?? ''
    expect(text).toMatch(/native SloNet/)
    expect(text).toMatch(/outside model/)

    // Banned per UX_FLOWS Feature #12 — scoped to this widget only.
    expect(text).not.toMatch(/epoch/i)
    expect(text).not.toMatch(/lora/i)
    expect(text).not.toMatch(/learning rate/i)
    expect(text).not.toMatch(/batch size/i)
    expect(text).not.toMatch(/hyperparameter/i)
    // No dead buttons: DPO and chat-train have no web-reachable entry yet.
    expect(text).not.toMatch(/\bdpo\b/i)
  })

  it('hands dataset and paste-text cards to the guided setup', () => {
    const { props } = renderChooser()

    // The guided setup starts on files — that card carries the "you are here" cue.
    expect(screen.getByRole('button', { name: /Train on a dataset/ })).toHaveAttribute(
      'aria-current',
      'true',
    )

    fireEvent.click(screen.getByRole('button', { name: /Train on a dataset/ }))
    expect(props.onPickDataset).toHaveBeenCalledTimes(1)
    expect(props.onPickText).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /Paste text/ }))
    expect(props.onPickText).toHaveBeenCalledTimes(1)
    expect(props.onPickDataset).toHaveBeenCalledTimes(1)
  })

  it('hands the quick card to the existing fast-train flow', () => {
    const { props } = renderChooser()

    fireEvent.click(screen.getByRole('button', { name: /Quick train/ }))
    expect(props.onPickQuick).toHaveBeenCalledTimes(1)
    expect(props.onPickDataset).not.toHaveBeenCalled()
  })

  it('starts from-chats training and streams progress into the card', async () => {
    mockCtl.startFromSessionsSloNet.mockResolvedValue(undefined)
    mockCtl.streamFromSessionsSloNet.mockReturnValue(
      sse([
        { status: 'working', message: 'Extracting chat pairs from sessions...', data: {} },
        {
          status: 'working',
          message: 'Learning from your chats...',
          data: { progress_percent: 42 },
        },
        { status: 'complete', message: 'Training complete', data: { progress_percent: 100 } },
      ]),
    )

    renderChooser()

    fireEvent.click(screen.getByRole('button', { name: /Train from your chats/ }))

    await waitFor(() => expect(mockCtl.startFromSessionsSloNet).toHaveBeenCalledWith({}))
    expect(await screen.findByText(/Finished/)).toBeTruthy()
    expect(mockCtl.cancelFromSessionsSloNet).not.toHaveBeenCalled()
  })

  it('surfaces a start failure instead of a dead button', async () => {
    mockCtl.startFromSessionsSloNet.mockRejectedValue(new Error('Training already in progress'))

    renderChooser()
    fireEvent.click(screen.getByRole('button', { name: /Train from your chats/ }))

    expect(await screen.findByText(/already in progress/)).toBeTruthy()
  })

  it('stops a running from-chats job from the card', async () => {
    let release: () => void = () => {}
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    async function* hanging() {
      yield {
        status: 'working',
        message: 'Learning from your chats...',
        data: { progress_percent: 10 },
      }
      await gate
      yield { status: 'complete', message: 'Training complete', data: { progress_percent: 100 } }
    }
    mockCtl.startFromSessionsSloNet.mockResolvedValue(undefined)
    mockCtl.streamFromSessionsSloNet.mockReturnValue(hanging())
    mockCtl.cancelFromSessionsSloNet.mockResolvedValue(undefined)

    renderChooser()
    fireEvent.click(screen.getByRole('button', { name: /Train from your chats/ }))

    expect(await screen.findByText(/Learning from your conversations/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(mockCtl.cancelFromSessionsSloNet).toHaveBeenCalled())
    expect(screen.queryByText(/Learning from your conversations/)).toBeNull()

    release()
    await Promise.resolve()
  })
})
