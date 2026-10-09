import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

vi.mock('@/components/composed/StatusBanner', () => ({
  StatusBanner: ({ message }: any) => <div role="alert">{message}</div>,
}))

const mockStore = vi.fn()

vi.mock('@/lib/memory-controller', () => ({
  memoryController: {
    store: (...a: any[]) => mockStore(...a),
  },
}))

import { MemoryAddForm } from './MemoryAddForm'
import { memoryController } from '@/lib/memory-controller'

afterEach(cleanup)

describe('MemoryAddForm', () => {
  const onAdded = vi.fn()
  const highlightItem = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockStore.mockResolvedValue({ stored: true })
    onAdded.mockResolvedValue(undefined)
  })

  it('renders without crashing', () => {
    render(<MemoryAddForm onAdded={onAdded} highlightItem={highlightItem} items={[]} />)
    expect(screen.getByLabelText('New memory fact')).toBeDefined()
  })

  it('renders save button disabled when content is empty', () => {
    render(<MemoryAddForm onAdded={onAdded} highlightItem={highlightItem} items={[]} />)
    const btn = screen.getByRole('button', { name: 'Save' })
    expect(btn.getAttribute('disabled')).not.toBeNull()
  })

  it('enables save button when content is entered', () => {
    render(<MemoryAddForm onAdded={onAdded} highlightItem={highlightItem} items={[]} />)
    const textarea = screen.getByLabelText('New memory fact')
    fireEvent.change(textarea, { target: { value: 'My fact' } })
    const btn = screen.getByRole('button', { name: 'Save' })
    expect(btn.getAttribute('disabled')).toBeNull()
  })

  it('calls memoryController.store on save and resets fields', async () => {
    render(<MemoryAddForm onAdded={onAdded} highlightItem={highlightItem} items={[]} />)
    const textarea = screen.getByLabelText('New memory fact')
    fireEvent.change(textarea, { target: { value: 'Remember this' } })
    const btn = screen.getByRole('button', { name: 'Save' })
    fireEvent.click(btn)

    await waitFor(() => {
      expect(mockStore).toHaveBeenCalledWith('Remember this', 'manual')
    })
    expect(onAdded).toHaveBeenCalled()
    expect(highlightItem).toHaveBeenCalledWith('Remember this', [])
  })

  it('shows error banner when store fails', async () => {
    mockStore.mockRejectedValueOnce(new Error('network'))
    render(<MemoryAddForm onAdded={onAdded} highlightItem={highlightItem} items={[]} />)
    const textarea = screen.getByLabelText('New memory fact')
    fireEvent.change(textarea, { target: { value: 'Fail fact' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeDefined()
    })
  })

  it('uses topic input value when provided', async () => {
    render(<MemoryAddForm onAdded={onAdded} highlightItem={highlightItem} items={[]} />)
    fireEvent.change(screen.getByLabelText('New memory fact'), { target: { value: 'Fact' } })
    fireEvent.change(screen.getByLabelText('Memory fact topic'), { target: { value: 'custom-topic' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => {
      expect(mockStore).toHaveBeenCalledWith('Fact', 'custom-topic')
    })
  })
})
