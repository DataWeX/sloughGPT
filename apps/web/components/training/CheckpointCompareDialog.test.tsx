// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockCompare = vi.fn()
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    compareCheckpoints: (a: string, b: string, prompt: string, tokens?: number) =>
      mockCompare(a, b, prompt, tokens),
  },
}))

import { CheckpointCompareDialog } from './CheckpointCompareDialog'

const checkpoints = [{ name: 'cp-1.soul' }, { name: 'cp-2.soul' }, { name: 'cp-3.soul' }]

const renderDialog = (overrides: Record<string, unknown> = {}) => {
  const addToast = vi.fn()
  const onClose = vi.fn()
  const view = render(
    <CheckpointCompareDialog
      open
      onClose={onClose}
      checkpoints={checkpoints}
      addToast={addToast}
      {...overrides}
    />,
  )
  return { ...view, addToast, onClose }
}

const typePrompt = (text: string) =>
  fireEvent.change(screen.getByLabelText('Comparison prompt'), { target: { value: text } })

describe('CheckpointCompareDialog', () => {
  afterEach(() => {
    cleanup()
    mockCompare.mockReset()
  })

  it('renders nothing when closed', () => {
    const { container } = render(
      <CheckpointCompareDialog
        open={false}
        onClose={vi.fn()}
        checkpoints={checkpoints}
        addToast={vi.fn()}
      />,
    )
    expect(container.textContent).toBe('')
  })

  it('offers every checkpoint on both sides', () => {
    renderDialog()
    const a = screen.getByLabelText('Checkpoint A') as HTMLSelectElement
    const b = screen.getByLabelText('Checkpoint B') as HTMLSelectElement
    expect([...a.options].map((o) => o.value)).toEqual(['cp-1.soul', 'cp-2.soul', 'cp-3.soul'])
    expect([...b.options].map((o) => o.value)).toEqual(['cp-1.soul', 'cp-2.soul', 'cp-3.soul'])
    expect(a.value).toBe('cp-1.soul')
    expect(b.value).toBe('cp-2.soul')
  })

  it('runs the same prompt against both selections', async () => {
    mockCompare.mockResolvedValue({
      a: { name: 'cp-1.soul', text: 'answer one' },
      b: { name: 'cp-3.soul', text: 'answer two' },
    })
    const { addToast } = renderDialog()

    fireEvent.change(screen.getByLabelText('Checkpoint B'), { target: { value: 'cp-3.soul' } })
    typePrompt('Say hello')
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }))

    await waitFor(() => {
      expect(mockCompare).toHaveBeenCalledWith('cp-1.soul', 'cp-3.soul', 'Say hello', undefined)
    })
    await waitFor(() => {
      expect(screen.getByText('answer one')).toBeDefined()
      expect(screen.getByText('answer two')).toBeDefined()
    })
    expect(screen.getByText('A · cp-1.soul')).toBeDefined()
    expect(addToast).not.toHaveBeenCalled()
  })

  it('needs a prompt before it runs', () => {
    renderDialog()
    const btn = screen.getByRole('button', { name: 'Compare' }) as HTMLButtonElement
    expect(btn.disabled).toBe(true)

    typePrompt('hello')
    expect((screen.getByRole('button', { name: 'Compare' }) as HTMLButtonElement).disabled).toBe(
      false,
    )
    expect(mockCompare).not.toHaveBeenCalled()
  })

  it('refuses to compare a checkpoint with itself', () => {
    renderDialog()
    fireEvent.change(screen.getByLabelText('Checkpoint B'), { target: { value: 'cp-1.soul' } })
    typePrompt('hello')
    expect((screen.getByRole('button', { name: 'Compare' }) as HTMLButtonElement).disabled).toBe(
      true,
    )
    expect(screen.getByText(/two different checkpoints/i)).toBeDefined()
  })

  it('reports a failed comparison as an error banner and toast', async () => {
    mockCompare.mockRejectedValue(new Error('load exploded'))
    const { addToast } = renderDialog()

    typePrompt('hello')
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }))

    await waitFor(() => {
      expect(addToast).toHaveBeenCalledWith(
        expect.stringContaining('Could not compare checkpoints'),
        'error',
      )
    })
    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeDefined()
      expect(screen.queryByText('answer one')).toBeNull()
    })
  })
})
