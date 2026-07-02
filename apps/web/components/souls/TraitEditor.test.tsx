// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import TraitEditor from './TraitEditor'

afterEach(cleanup)

const defaultWeights: Record<string, Record<string, number>> = {
  personality: { warmth: 0.7, empathy: 0.6, openness: 0.5 },
  cognition: { creativity: 0.8, curiosity: 0.4, pattern_recognition: 0.3 },
  emotion: { resilience: 0.5, optimism: 0.6, patience: 0.2 },
}

/** StrictMode renders components twice — use getAllByText and pick first match */
function firstByText(text: string | RegExp): HTMLElement {
  return screen.getAllByText(text)[0]
}

/** Find a slider whose aria-label starts with the given label (StrictMode-safe, value-independent). */
function sliderByLabel(label: string): HTMLElement {
  return screen.getAllByRole('slider').find(s => (s.getAttribute('aria-label') || '').startsWith(label))!
}

/** Set a native <input type="range"> slider to a target value. Uses the native
 *  HTMLInputElement.value setter (bypasses JSDOM limitations) + fireEvent.change
 *  to trigger React's onChange handler. */
function setSlider(label: string, value: number) {
  const slider = sliderByLabel(label) as HTMLInputElement
  const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!
  nativeInputValueSetter.call(slider, String(value))
  fireEvent.change(slider)
}

describe('TraitEditor', () => {
  it('renders archetype badge', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    expect(firstByText('Archetype')).toBeDefined()
    // optimism=0.6 meets threshold → The Optimist
    expect(firstByText('The Optimist')).toBeDefined()
  })

  it('renders all three groups', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    expect(screen.getAllByText(/Personality/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Cognition/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Emotion/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders trait labels', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Creativity').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Resilience').length).toBeGreaterThanOrEqual(1)
  })

  it('shows trait values as percentages', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    expect(screen.getAllByText('70').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('80').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('20').length).toBeGreaterThanOrEqual(1)
  })

  it('Save button is disabled when no changes', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    expect(screen.getAllByText('Save')[0].closest('button')).toBeDisabled()
  })

  it('Save button enables after slider change', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    setSlider('Warmth: 70', 90)
    expect(screen.getAllByText('Save')[0].closest('button')).not.toBeDisabled()
  })

  it('archetype updates after slider change', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    expect(firstByText('The Optimist')).toBeDefined()
    setSlider('Warmth: 70', 5)
    setSlider('Empathy: 60', 5)
    setSlider('Optimism: 60', 5)
    expect(firstByText('The Balanced')).toBeDefined()
  })

  it('calls onSave with updated weights', () => {
    const onSave = vi.fn()
    render(<TraitEditor traitWeights={defaultWeights} onSave={onSave} onReset={vi.fn()} />)
    setSlider('Warmth: 70', 90)
    fireEvent.click(screen.getAllByText('Save')[0])
    expect(onSave).toHaveBeenCalledTimes(1)
    const saved = onSave.mock.calls[0][0] as Record<string, Record<string, number>>
    expect(saved.personality.warmth).toBe(0.9)
  })

  it('calls onReset on Reset click', () => {
    const onReset = vi.fn()
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={onReset} />)
    setSlider('Warmth: 70', 90)
    fireEvent.click(screen.getAllByText('Reset')[0])
    expect(onReset).toHaveBeenCalledTimes(1)
  })

  it('displays archetype description', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    expect(firstByText(/Upbeat|encouraging|bright/)).toBeDefined()
  })

  it('all trait sliders have aria-labels', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    const sliders = screen.getAllByRole('slider')
    expect(sliders.length).toBeGreaterThanOrEqual(9)
    sliders.forEach(s => {
      expect(s.getAttribute('aria-label')).toBeTruthy()
    })
  })

  it('dirty state resets after saving', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    setSlider('Warmth: 70', 90)
    fireEvent.click(screen.getAllByText('Save')[0])
    expect(screen.getAllByText('Save')[0].closest('button')).toBeDisabled()
  })

  it('Reset restores original values from props', () => {
    render(<TraitEditor traitWeights={defaultWeights} onSave={vi.fn()} onReset={vi.fn()} />)
    setSlider('Warmth: 70', 90)
    fireEvent.click(screen.getAllByText('Reset')[0])
    expect(screen.getAllByText('70').length).toBeGreaterThanOrEqual(1)
  })
})
