import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { StepContext, STEP_HINTS } from './StepContext'

afterEach(() => cleanup())

describe('STEP_HINTS', () => {
  it('defines hints for all training steps', () => {
    expect(STEP_HINTS['data-selector']).toBeDefined()
    expect(STEP_HINTS['configure-method']).toBeDefined()
    expect(STEP_HINTS['train-start']).toBeDefined()
    expect(STEP_HINTS['results-checkpoint']).toBeDefined()
  })

  it('each hint has title and content', () => {
    Object.values(STEP_HINTS).forEach(hint => {
      expect(hint.title).toBeTruthy()
      expect(hint.content).toBeTruthy()
    })
  })
})

describe('StepContext', () => {
  it('renders children', () => {
    render(
      <StepContext hintKey="data-selector">
        <span>Select data</span>
      </StepContext>
    )
    expect(screen.getAllByText('Select data').length).toBeGreaterThan(0)
  })

  it('does not show hint for invalid key', () => {
    render(
      <StepContext hintKey="nonexistent">
        <span>Content</span>
      </StepContext>
    )
    expect(screen.getAllByText('Content').length).toBeGreaterThan(0)
    expect(screen.queryByRole('button', { name: /help/i })).not.toBeInTheDocument()
  })

  it('shows info button for valid key', () => {
    render(
      <StepContext hintKey="data-selector">
        <span>Select</span>
      </StepContext>
    )
    expect(screen.getByRole('button', { name: /help.*dataset selection/i })).toBeInTheDocument()
  })

  it('opens hint popup on click', () => {
    render(
      <StepContext hintKey="data-selector">
        <span>Select</span>
      </StepContext>
    )
    fireEvent.click(screen.getByRole('button', { name: /help/i }))
    expect(screen.getByText('Dataset selection')).toBeInTheDocument()
    expect(screen.getByText(/Choose an existing dataset/)).toBeInTheDocument()
  })

  it('shows tip when available', () => {
    render(
      <StepContext hintKey="data-selector">
        <span>Select</span>
      </StepContext>
    )
    fireEvent.click(screen.getByRole('button', { name: /help/i }))
    expect(screen.getByText(/Start with a small dataset/)).toBeInTheDocument()
  })

  it('closes on close button click', async () => {
    render(
      <StepContext hintKey="data-selector">
        <span>Select</span>
      </StepContext>
    )
    fireEvent.click(screen.getByRole('button', { name: /help/i }))
    fireEvent.click(screen.getByRole('button', { name: /close help/i }))
    await waitFor(() => {
      expect(screen.queryByText('Dataset selection')).not.toBeInTheDocument()
    })
  })

  it('exports STEP_HINTS', () => {
    expect(STEP_HINTS).toBeDefined()
    expect(typeof STEP_HINTS).toBe('object')
  })
})
