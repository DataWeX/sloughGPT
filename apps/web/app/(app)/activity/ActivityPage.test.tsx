import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

const { mockStatus, mockDataset, mockTrain, mockPredict, mockRecordData, mockDeleteAll, mockAddToast, mockStartActivityTraining, mockListJobs } = vi.hoisted(() => ({
  mockStatus: vi.fn(),
  mockDataset: vi.fn(),
  mockTrain: vi.fn(),
  mockPredict: vi.fn(),
  mockRecordData: vi.fn(),
  mockDeleteAll: vi.fn(),
  mockAddToast: vi.fn(),
  mockStartActivityTraining: vi.fn(),
  mockListJobs: vi.fn(),
}))

vi.mock('@/lib/activity-controller', () => ({
  activityController: {
    status: mockStatus,
    dataset: mockDataset,
    train: mockTrain,
    predict: mockPredict,
    recordData: mockRecordData,
    deleteAll: mockDeleteAll,
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))
vi.mock('@/lib/controllers', () => ({
  trainingJobsController: {
    startActivityTraining: mockStartActivityTraining,
    list: mockListJobs,
  },
}))

import ActivityPage from './page'

afterEach(() => { cleanup() })
beforeEach(() => {
  vi.clearAllMocks()
  mockStatus.mockResolvedValue({
    model_loaded: true,
    num_recordings: 10,
    num_labels: 3,
    activities: ['stationary', 'walking', 'running'],
    device: 'cpu',
  })
  mockDataset.mockResolvedValue({
    recordings: [
      { id: 1, path: 'r1.npz', samples: 128, label: 0, activity: 'stationary' },
      { id: 2, path: 'r2.npz', samples: 128, label: 1, activity: 'walking' },
    ],
    total: 2,
  })
})

describe('ActivityPage', () => {

  it('renders page title', async () => {
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Activity Recognition').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows loading spinner initially', () => {
    mockStatus.mockReturnValue(new Promise(() => {}))
    mockDataset.mockReturnValue(new Promise(() => {}))
    render(<ActivityPage />)
    expect(screen.getByText('System Status')).toBeTruthy()
  })

  it('displays status stats after loading', async () => {
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('10').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows "Server offline" when status fails', async () => {
    mockStatus.mockRejectedValueOnce(new Error('offline'))
    mockDataset.mockResolvedValue({ recordings: [], total: 0 })
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getByText('Server offline')).toBeTruthy()
    })
  })

  it('displays recorded dataset entries', async () => {
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('stationary').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('walking').length).toBeGreaterThanOrEqual(1)
  })



  it('shows empty dataset message when no recordings', async () => {
    mockDataset.mockResolvedValue({ recordings: [], total: 0 })
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getByText(/No recordings yet/)).toBeTruthy()
    })
  })

  it('shows activity chips for data collection', async () => {
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getAllByText('stationary').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('walking').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('running').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('shaking').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('driving').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('cycling').length).toBeGreaterThanOrEqual(1)
  })

  it('simulates recording and uploads data', async () => {
    mockRecordData.mockResolvedValue({ id: 99, path: 'r99.npz', samples: 128 })
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getByText('Simulate sample')).toBeTruthy()
    })
    await act(async () => { fireEvent.click(screen.getByText('Simulate sample')) })
    await waitFor(() => {
      expect(mockRecordData).toHaveBeenCalled()
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('Saved recording'),
        'success'
      )
    })
  })

  it('trains classifier', async () => {
    mockStartActivityTraining.mockResolvedValue({ job_id: 'activity_1', status: 'queued' })
    mockListJobs.mockResolvedValue([{
      id: 'activity_1',
      status: 'completed',
      metrics: { val_accuracy: 0.85, num_samples: 10 },
      epochs_completed: 30,
    }])
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getByText('Train')).toBeTruthy()
    })
    await act(async () => { fireEvent.click(screen.getByText('Train')) })
    await waitFor(() => {
      expect(screen.getByText(/85.0%/)).toBeTruthy()
      expect(mockStartActivityTraining).toHaveBeenCalled()
    })
  })

  it('predicts from buffer', async () => {
    mockRecordData.mockResolvedValue({ id: 1, path: 'r1.npz', samples: 128 })
    mockPredict.mockResolvedValue({
      activity: 'walking',
      class_id: 1,
      confidence: 0.92,
      probabilities: [0.05, 0.92, 0.03],
    })
    render(<ActivityPage />)
    await waitFor(() => { expect(screen.getByText('Simulate sample')).toBeTruthy() })

    // First simulate to populate buffer
    await act(async () => { fireEvent.click(screen.getByText('Simulate sample')) })
    await waitFor(() => { expect(mockRecordData).toHaveBeenCalled() })

    // Then predict
    await act(async () => { fireEvent.click(screen.getByText('Predict from buffered data')) })
    await waitFor(() => {
      expect(screen.getAllByText('walking').length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('92% confidence')).toBeTruthy()
    })
  })

  it('shows error toast on train failure', async () => {
    mockStartActivityTraining.mockRejectedValueOnce(new Error('train failed'))
    render(<ActivityPage />)
    await waitFor(() => { expect(screen.getByText('Train')).toBeTruthy() })
    await act(async () => { fireEvent.click(screen.getByText('Train')) })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('Training failed'),
        'error'
      )
    })
  })

  it('delete all clears dataset', async () => {
    mockDeleteAll.mockResolvedValue({ deleted: 5 })
    render(<ActivityPage />)
    await waitFor(() => { expect(screen.getByText('Delete all data')).toBeTruthy() })
    await act(async () => { fireEvent.click(screen.getByText('Delete all data')) })
    await waitFor(() => {
      expect(mockDeleteAll).toHaveBeenCalled()
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('Deleted 5'),
        'success'
      )
    })
  })

  it('quick-fill buttons exist for each activity', async () => {
    render(<ActivityPage />)
    await waitFor(() => {
      expect(screen.getByText(/Quick-fill dataset/)).toBeTruthy()
    })
    expect(screen.getByText('+stationary')).toBeTruthy()
    expect(screen.getByText('+walking')).toBeTruthy()
    expect(screen.getByText('+running')).toBeTruthy()
    expect(screen.getByText('+shaking')).toBeTruthy()
    expect(screen.getByText('+driving')).toBeTruthy()
    expect(screen.getByText('+cycling')).toBeTruthy()
  })
})
