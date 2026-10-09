import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Slider: ({ label, value, onValueChange, ...props }: any) => (
    <input
      type="range"
      aria-label={label}
      value={value?.[0] ?? 0}
      onChange={(e) => onValueChange?.([parseFloat(e.target.value)])}
      {...props}
    />
  ),
  IconEdit: (props: any) => <span {...props} />,
}))

vi.mock('@/components/composed/StatusBanner', () => ({
  StatusBanner: ({ message }: any) => <div role="alert">{message}</div>,
}))

const mockUpdate = vi.fn()

vi.mock('@/lib/memory-controller', () => ({
  memoryController: {
    update: (...a: any[]) => mockUpdate(...a),
  },
}))

import { MemoryEditForm } from './MemoryEditForm'
import type { MemoryItem } from '@/lib/memory-controller'

const mockItem: MemoryItem = {
  id: 'item-1',
  content: 'Original content',
  topic: 'test-topic',
  source: 'manual',
  url: '',
  timestamp: Date.now(),
  importance: 0.8,
  score: 0,
}

afterEach(cleanup)

describe('MemoryEditForm', () => {
  const onSaved = vi.fn()
  const onCancelled = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockUpdate.mockResolvedValue({ updated: 1 })
    onSaved.mockResolvedValue(undefined)
  })

  it('renders without crashing', () => {
    render(<MemoryEditForm item={mockItem} onSaved={onSaved} onCancelled={onCancelled} />)
    expect(screen.getByLabelText('Edit memory fact text')).toBeDefined()
  })

  it('pre-fills fields from item', () => {
    render(<MemoryEditForm item={mockItem} onSaved={onSaved} onCancelled={onCancelled} />)
    expect((screen.getByLabelText('Edit memory fact text') as HTMLTextAreaElement).value).toBe('Original content')
    expect((screen.getByLabelText('Edit memory fact topic') as HTMLInputElement).value).toBe('test-topic')
  })

  it('calls memoryController.update on save with edited content', async () => {
    render(<MemoryEditForm item={mockItem} onSaved={onSaved} onCancelled={onCancelled} />)
    const textarea = screen.getByLabelText('Edit memory fact text')
    fireEvent.change(textarea, { target: { value: 'Updated content' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => {
      expect(mockUpdate).toHaveBeenCalledWith('item-1', 'Updated content', 'test-topic', 0.8)
    })
    expect(onSaved).toHaveBeenCalled()
  })

  it('calls onCancelled when cancel is clicked', () => {
    render(<MemoryEditForm item={mockItem} onSaved={onSaved} onCancelled={onCancelled} />)
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancelled).toHaveBeenCalled()
  })

  it('shows error on duplicate', async () => {
    mockUpdate.mockResolvedValueOnce({ updated: 0, duplicate: true })
    render(<MemoryEditForm item={mockItem} onSaved={onSaved} onCancelled={onCancelled} />)
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeDefined()
    })
  })

  it('disables save button when content is empty', () => {
    render(<MemoryEditForm item={mockItem} onSaved={onSaved} onCancelled={onCancelled} />)
    const textarea = screen.getByLabelText('Edit memory fact text')
    fireEvent.change(textarea, { target: { value: '' } })
    const btn = screen.getByRole('button', { name: 'Save' })
    expect(btn.getAttribute('disabled')).not.toBeNull()
  })
})
