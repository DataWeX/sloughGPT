// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockGetStatus, mockStart, mockStop,
} = vi.hoisted(() => ({
  mockGetStatus: vi.fn(), mockStart: vi.fn(), mockStop: vi.fn(),
}))

vi.mock('@/lib/self-train-controller', () => ({
  selfTrainController: {
    getStatus: (...a: unknown[]) => mockGetStatus(...a),
    start: (...a: unknown[]) => mockStart(...a),
    stop: (...a: unknown[]) => mockStop(...a),
  },
}))

import { SelfTrainCard } from './SelfTrainCard'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGetStatus.mockResolvedValue({ status: 'not_started', history: [], pid: null })
  mockStart.mockResolvedValue({ status: 'started' })
  mockStop.mockResolvedValue({ status: 'stopped' })
})

describe('SelfTrainCard — initial load', () => {
  it('renders card title', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getByText('Self-Train')).toBeTruthy())
  })

  it('renders model input', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getAllByPlaceholderText(/model/i).length).toBeGreaterThanOrEqual(1))
  })

  it('renders temperature input', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getAllByPlaceholderText(/temperature/i).length).toBeGreaterThanOrEqual(1))
  })

  it('renders start button when idle', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getAllByText('Start').length).toBeGreaterThanOrEqual(1))
  })

  it('renders stop button when idle', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getAllByText('Stop').length).toBeGreaterThanOrEqual(1))
  })

  it('renders train indefinitely checkbox', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getByText(/train indefinitely/i)).toBeTruthy())
  })
})

describe('SelfTrainCard — running state', () => {
  it('shows running state with PID', async () => {
    mockGetStatus.mockResolvedValue({ status: 'running', history: [], pid: 12345 })
    render(<SelfTrainCard />)
    await waitFor(() => {
      expect(screen.getByText('Running...')).toBeTruthy()
      expect(screen.getByText(/PID 12345/)).toBeTruthy()
    })
  })

  it('disables start button when running', async () => {
    mockGetStatus.mockResolvedValue({ status: 'running', history: [], pid: 1 })
    render(<SelfTrainCard />)
    await waitFor(() => {
      const startBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('Running'))
      expect(startBtn?.getAttribute('disabled')).not.toBeNull()
    })
  })

  it('enables stop button when running', async () => {
    mockGetStatus.mockResolvedValue({ status: 'running', history: [], pid: 1 })
    render(<SelfTrainCard />)
    await waitFor(() => {
      const stopBtn = screen.getAllByRole('button').find(b => b.textContent === 'Stop')
      expect(stopBtn?.getAttribute('disabled')).toBeNull()
    })
  })
})

describe('SelfTrainCard — history display', () => {
  it('shows training history when available', async () => {
    mockGetStatus.mockResolvedValue({
      status: 'running',
      history: ['Step 1: loss=2.5', 'Step 2: loss=2.1', 'Step 3: loss=1.8'],
      pid: 1,
    })
    render(<SelfTrainCard />)
    await waitFor(() => {
      expect(screen.getByText('Step 1: loss=2.5')).toBeTruthy()
      expect(screen.getByText('Step 3: loss=1.8')).toBeTruthy()
    })
  })

  it('limits history to last 10 entries', async () => {
    const history = Array.from({ length: 15 }, (_, i) => `Step ${i + 1}`)
    mockGetStatus.mockResolvedValue({ status: 'running', history, pid: 1 })
    render(<SelfTrainCard />)
    await waitFor(() => {
      expect(screen.queryByText('Step 1')).toBeNull()
      expect(screen.getByText('Step 15')).toBeTruthy()
    })
  })
})

describe('SelfTrainCard — start flow', () => {
  it('calls start with default params', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => { expect(screen.getAllByText('Start').length).toBeGreaterThanOrEqual(1) })

    const startBtn = screen.getAllByRole('button').find(b => b.textContent === 'Start')
    if (startBtn) {
      await act(async () => { fireEvent.click(startBtn) })
      await waitFor(() => {
        expect(mockStart).toHaveBeenCalled()
      })
    }
  })

  it('passes model and temperature values', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/model/i).length).toBeGreaterThanOrEqual(1) })

    const modelInput = screen.getAllByPlaceholderText(/model/i)[0]
    const tempInput = screen.getAllByPlaceholderText(/temperature/i)[0]
    fireEvent.change(modelInput, { target: { value: 'gpt2' } })
    fireEvent.change(tempInput, { target: { value: '0.5' } })

    const startBtn = screen.getAllByRole('button').find(b => b.textContent === 'Start')
    if (startBtn) {
      await act(async () => { fireEvent.click(startBtn) })
      await waitFor(() => {
        expect(mockStart).toHaveBeenCalledWith(expect.objectContaining({
          model: 'gpt2',
          temperature: 0.5,
        }))
      })
    }
  })

  it('passes forever flag when checked', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => { expect(screen.getByText(/train indefinitely/i)).toBeTruthy() })

    const checkbox = screen.getByRole('checkbox')
    fireEvent.click(checkbox)

    const startBtn = screen.getAllByRole('button').find(b => b.textContent === 'Start')
    if (startBtn) {
      await act(async () => { fireEvent.click(startBtn) })
      await waitFor(() => {
        expect(mockStart).toHaveBeenCalledWith(expect.objectContaining({ forever: true }))
      })
    }
  })

  it('shows error on start failure', async () => {
    mockStart.mockRejectedValue(new Error('Model not found'))
    render(<SelfTrainCard />)
    await waitFor(() => { expect(screen.getAllByText('Start').length).toBeGreaterThanOrEqual(1) })

    const startBtn = screen.getAllByRole('button').find(b => b.textContent === 'Start')
    if (startBtn) {
      await act(async () => { fireEvent.click(startBtn) })
      await waitFor(() => {
        expect(screen.getByText('Model not found')).toBeTruthy()
      })
    }
  })
})

describe('SelfTrainCard — stop flow', () => {
  it('calls stop when stop button clicked', async () => {
    mockGetStatus.mockResolvedValue({ status: 'running', history: [], pid: 1 })
    render(<SelfTrainCard />)
    await waitFor(() => { expect(screen.getByText('Stop')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Stop')) })
    await waitFor(() => {
      expect(mockStop).toHaveBeenCalled()
    })
  })
})

describe('SelfTrainCard — offline handling', () => {
  it('handles status fetch failure gracefully', async () => {
    mockGetStatus.mockRejectedValue(new Error('offline'))
    render(<SelfTrainCard />)
    await waitFor(() => {
      expect(screen.getByText('Self-Train')).toBeTruthy()
    })
  })
})
