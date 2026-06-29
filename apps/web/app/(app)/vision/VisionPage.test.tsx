// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockAnalyzeImage, mockTrainImage, mockGenerateImage, mockGetTrainingReport, mockResetModel } = vi.hoisted(() => ({
  mockAnalyzeImage: vi.fn(),
  mockTrainImage: vi.fn(),
  mockGenerateImage: vi.fn(),
  mockGetTrainingReport: vi.fn(),
  mockResetModel: vi.fn(),
}))

const { mockListCheckpoints, mockLoadCheckpoint, mockDeleteCheckpoint } = vi.hoisted(() => ({
  mockListCheckpoints: vi.fn(),
  mockLoadCheckpoint: vi.fn(),
  mockDeleteCheckpoint: vi.fn(),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
  useParams: () => ({}),
}))

vi.mock('@/lib/multimodal-controller', () => ({
  multimodalController: {
    analyzeImage: mockAnalyzeImage,
    trainImage: mockTrainImage,
    generateImage: mockGenerateImage,
    getTrainingReport: mockGetTrainingReport,
    resetModel: mockResetModel,
  },
}))

vi.mock('@/lib/visual-controller', () => ({
  visualController: {
    listCheckpoints: mockListCheckpoints,
    loadCheckpoint: mockLoadCheckpoint,
    deleteCheckpoint: mockDeleteCheckpoint,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel ? sel({ addToast: vi.fn() }) : { addToast: vi.fn() },
}))

import VisionPage from './page'

const mockReport = { images_learned: 42, vocab_size: 1000, caption_history: [], accuracy_history: [], mean_accuracy: 75, last_accuracy: 80 }

beforeEach(() => {
  vi.clearAllMocks()
  mockGetTrainingReport.mockResolvedValue(mockReport)
  mockListCheckpoints.mockResolvedValue([])
})

afterEach(cleanup)

describe('VisionPage', () => {
  it('renders header and tabs', async () => {
    render(<VisionPage />)
    await waitFor(() => expect(screen.getByText('Vision Studio')).toBeDefined())
    expect(screen.getByText('Analyze')).toBeDefined()
    expect(screen.getByText('Supervised Train')).toBeDefined()
    expect(screen.getByText('Generate')).toBeDefined()
  })

  it('switches to Train tab', async () => {
    render(<VisionPage />)
    await waitFor(() => expect(screen.getByText('Supervised Train')).toBeDefined())
    fireEvent.click(screen.getByText('Supervised Train'))
    await waitFor(() => expect(screen.getByText('Train with label')).toBeDefined())
  })

  it('shows drag-drop overlay on Train tab dragEnter', async () => {
    render(<VisionPage />)
    await waitFor(() => expect(screen.getByText('Supervised Train')).toBeDefined())
    fireEvent.click(screen.getByText('Supervised Train'))
    await waitFor(() => expect(screen.getByText('Train with label')).toBeDefined())
    const trainDropText = screen.getByText(/click to select for training/)
    expect(trainDropText).toBeDefined()
    const trainZone = trainDropText.closest('[class*="border-2"]') as HTMLElement
    expect(trainZone).not.toBeNull()
    fireEvent.dragOver(trainZone)
    await waitFor(() => expect(screen.getByText('Drop image here')).toBeDefined())
  })

  it('drops image on Train tab sets preview without analyzing', async () => {
    render(<VisionPage />)
    await waitFor(() => expect(screen.getByText('Supervised Train')).toBeDefined())
    fireEvent.click(screen.getByText('Supervised Train'))
    await waitFor(() => expect(screen.getByText('Train with label')).toBeDefined())
    const trainDropText = screen.getByText(/click to select for training/)
    expect(trainDropText).toBeDefined()
    const trainZone = trainDropText.closest('[class*="border-2"]') as HTMLElement
    expect(trainZone).not.toBeNull()
    const file = new File([''], 'test.png', { type: 'image/png' })
    Object.defineProperty(file, 'size', { value: 1024 })
    fireEvent.drop(trainZone, { dataTransfer: { files: [file] } })
    await new Promise(r => setTimeout(r, 50))
    expect(mockAnalyzeImage).not.toHaveBeenCalled()
  })

  it('switches to Generate tab', async () => {
    render(<VisionPage />)
    await waitFor(() => expect(screen.getByText('Generate')).toBeDefined())
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getByPlaceholderText('Describe an image to generate...')).toBeDefined())
  })

  it('switches to History tab', async () => {
    render(<VisionPage />)
    await waitFor(() => expect(screen.getByText('History')).toBeDefined())
    fireEvent.click(screen.getByText('History'))
    await waitFor(() => expect(screen.getByText('Images learned')).toBeDefined())
  })
})
