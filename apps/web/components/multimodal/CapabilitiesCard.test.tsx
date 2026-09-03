import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import CapabilitiesCard from './CapabilitiesCard'

afterEach(() => cleanup())

const fullCaps = {
  speech_to_text: true, image_caption: true, vision_model: 'gpt-4v',
  speech_model: 'whisper', trained: true, images_learned: 10,
  replay_buffer_size: 500, learning_method: 'DPO', status: 'ready',
}

describe('CapabilitiesCard', () => {
  it('renders all capability badges', () => {
    render(<CapabilitiesCard caps={fullCaps} />)
    expect(screen.getAllByText('Speech-to-text').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Image captioning').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Vision model').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Speech model').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Trained').length).toBeGreaterThanOrEqual(1)
  })
  it('shows stats', () => {
    render(<CapabilitiesCard caps={fullCaps} />)
    expect(screen.getAllByText('10').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('500 items').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('DPO').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('ready').length).toBeGreaterThanOrEqual(1)
  })
  it('shows empty state when caps null', () => {
    render(<CapabilitiesCard caps={null} />)
    expect(screen.getAllByText('0').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(1)
  })
  it('renders heading', () => {
    render(<CapabilitiesCard caps={null} />)
    expect(screen.getAllByText('Capabilities').length).toBeGreaterThanOrEqual(1)
  })
})
