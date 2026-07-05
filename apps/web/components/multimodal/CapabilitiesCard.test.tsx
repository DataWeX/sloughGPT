import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import CapabilitiesCard from './CapabilitiesCard'

describe('CapabilitiesCard', () => {
  afterEach(cleanup)

  const caps = {
    speech_to_text: true,
    image_caption: false,
    vision_model: 'vision-cnn',
    speech_model: null,
    trained: true,
    images_learned: 42,
    replay_buffer_size: 128,
    learning_method: 'contrastive',
    background_job_running: false,
    status: 'ready',
  }

  it('renders capabilities as badges', () => {
    render(<CapabilitiesCard caps={caps} />)
    expect(screen.getByText('Speech-to-text')).toBeDefined()
    expect(screen.getByText('Image captioning')).toBeDefined()
    expect(screen.getByText('Vision model')).toBeDefined()
    expect(screen.getByText('Trained')).toBeDefined()
  })

  it('renders KPI grid values', () => {
    render(<CapabilitiesCard caps={caps} />)
    expect(screen.getByText('42')).toBeDefined()
    expect(screen.getByText('128 items')).toBeDefined()
    expect(screen.getByText('contrastive')).toBeDefined()
    expect(screen.getByText('ready')).toBeDefined()
  })

  it('handles null caps', () => {
    render(<CapabilitiesCard caps={null} />)
    expect(screen.getByText('Images learned')).toBeDefined()
    expect(screen.getByText('0')).toBeDefined()
  })
})
