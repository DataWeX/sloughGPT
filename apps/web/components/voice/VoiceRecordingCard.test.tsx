// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const STORAGE_KEY = 'sloughgpt-voice-recordings'

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn((key: string) => {
      const raw = localStorage.getItem(key)
      return Promise.resolve(raw ? JSON.parse(raw) : undefined)
    }),
    setKV: vi.fn((key: string, value: unknown) => {
      localStorage.setItem(key, JSON.stringify(value))
      return Promise.resolve()
    }),
    deleteKV: vi.fn((key: string) => {
      localStorage.removeItem(key)
      return Promise.resolve()
    }),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

const mockStream = {
  getTracks: vi.fn().mockReturnValue([{ stop: vi.fn() }]),
}

const mockMediaRecorder = {
  start: vi.fn(),
  stop: vi.fn(),
  state: 'idle',
  ondataavailable: null as any,
  onstop: null as any,
}

Object.defineProperty(navigator, 'mediaDevices', {
  value: { getUserMedia: vi.fn().mockResolvedValue(mockStream) },
  writable: true,
})

vi.stubGlobal('MediaRecorder', vi.fn().mockImplementation(() => {
  mockMediaRecorder.state = 'recording'
  return mockMediaRecorder
}))

vi.stubGlobal('AudioContext', vi.fn().mockImplementation(() => ({
  createMediaStreamSource: vi.fn().mockReturnValue({
    connect: vi.fn(),
  }),
  createAnalyser: vi.fn().mockReturnValue({
    fftSize: 256,
    getByteTimeDomainData: vi.fn(),
  }),
  close: vi.fn(),
})))

vi.stubGlobal('Audio', vi.fn().mockImplementation(() => ({
  play: vi.fn(),
})))

import { VoiceRecordingCard } from './VoiceRecordingCard'

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  cleanup()
})

describe('VoiceRecordingCard', () => {
  it('renders the component', () => {
    render(<VoiceRecordingCard />)
    expect(screen.getByTestId('voice-recording')).toBeTruthy()
  })

  it('shows idle state with record button', () => {
    render(<VoiceRecordingCard />)
    expect(screen.getByText('Voice Recording')).toBeTruthy()
  })

  it('starts recording on click', async () => {
    render(<VoiceRecordingCard />)
    const card = document.querySelector('[data-testid="voice-recording"]')!
    const recordBtn = card.querySelector('.rounded-full')!
    expect(recordBtn).toBeTruthy()
    fireEvent.click(recordBtn)
    await waitFor(() => {
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalled()
    })
  })

  it('shows empty state when no recordings', () => {
    render(<VoiceRecordingCard />)
    expect(screen.queryByText('Play')).toBeNull()
  })

  it('loads existing recordings from localStorage', async () => {
    const recordings = [
      {
        id: 'rec-1',
        url: 'blob:mock',
        duration: 5000,
        label: 'Test Recording',
        timestamp: Date.now(),
      },
    ]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(recordings))
    render(<VoiceRecordingCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Test Recording').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows duration format', async () => {
    const recordings = [
      {
        id: 'rec-1',
        url: 'blob:mock',
        duration: 65000,
        label: 'Long Recording',
        timestamp: Date.now(),
      },
    ]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(recordings))
    render(<VoiceRecordingCard />)
    await waitFor(() => {
      expect(screen.getAllByText('1:05').length).toBeGreaterThanOrEqual(1)
    })
  })
})
