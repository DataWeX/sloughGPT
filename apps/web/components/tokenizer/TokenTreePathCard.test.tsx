import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockPath: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    path: mocks.mockPath,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mocks.mockAddToast }) => unknown) =>
    selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Input: (props: any) => <input {...props} />,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  IconSearch: () => <span />,
}))

import { TokenTreePathCard } from './TokenTreePathCard'

const TRACE = {
  steps: [
    { remaining: 'the</w>', token: 'the</w>', id: 3, consumed: 7 },
    { remaining: ' quick</w>', token: ' quick</w>', id: 27, consumed: 10 },
  ],
  ids: [3, 27],
}

describe('TokenTreePathCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the text input with a default', () => {
    render(<TokenTreePathCard />)
    expect(screen.getByLabelText('Text to trace')).toBeDefined()
    expect((screen.getByLabelText('Text to trace') as HTMLInputElement).value).toBe('the quick brown fox')
  })

  it('traces a path and renders steps with tokens and ids', async () => {
    mocks.mockPath.mockResolvedValue(TRACE)
    render(<TokenTreePathCard />)

    fireEvent.change(screen.getByLabelText('Text to trace'), { target: { value: 'the quick' } })
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    await waitFor(() => expect(mocks.mockPath).toHaveBeenCalledWith('the quick'))

    expect(screen.getByText('#1')).toBeDefined()
    expect(screen.getByText('the</w>')).toBeDefined()
    expect(screen.getByText('id 3')).toBeDefined()
    expect(screen.getByText('[3, 27]')).toBeDefined()
  })

  it('disables the button for a blank text', () => {
    render(<TokenTreePathCard />)
    fireEvent.change(screen.getByLabelText('Text to trace'), { target: { value: '   ' } })
    expect((screen.getByRole('button', { name: /^Trace$/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('traces on Enter key', async () => {
    mocks.mockPath.mockResolvedValue(TRACE)
    render(<TokenTreePathCard />)
    fireEvent.keyDown(screen.getByLabelText('Text to trace'), { key: 'Enter' })
    await waitFor(() => expect(mocks.mockPath).toHaveBeenCalledWith('the quick brown fox'))
  })

  it('shows a toast when tracing fails', async () => {
    mocks.mockPath.mockRejectedValue(new Error('boom'))
    render(<TokenTreePathCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Could not trace the token path', 'error'))
  })

  it('renders step numbers as #1, #2, etc.', async () => {
    mocks.mockPath.mockResolvedValue(TRACE)
    render(<TokenTreePathCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    await waitFor(() => expect(screen.getByText('#1')).toBeDefined())
    expect(screen.getByText('#2')).toBeDefined()
  })

  it('shows consumed character counts', async () => {
    mocks.mockPath.mockResolvedValue(TRACE)
    render(<TokenTreePathCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    await waitFor(() => expect(screen.getByText('+7')).toBeDefined())
    expect(screen.getByText('+10')).toBeDefined()
  })

  it('shows tracing... while loading', async () => {
    let resolvePromise: any
    mocks.mockPath.mockImplementation(() => new Promise(r => { resolvePromise = r }))
    render(<TokenTreePathCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    expect(screen.getByText('Tracing...')).toBeDefined()
    resolvePromise(TRACE)
    await waitFor(() => expect(screen.getByText('Trace')).toBeDefined())
  })

  it('renders the CardTitle', () => {
    render(<TokenTreePathCard />)
    expect(screen.getByText('Token Path Explorer')).toBeDefined()
  })
})
