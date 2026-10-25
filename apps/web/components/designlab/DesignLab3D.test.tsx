// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import DesignLab3D, { MATERIAL_KINDS } from './DesignLab3D'

afterEach(cleanup)

describe('DesignLab3D (dev-only 3D dock showcase)', () => {
  it('renders the concept header and dev-only tag', () => {
    render(<DesignLab3D />)
    expect(screen.getByTestId('designlab-3d')).toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('3D dock')
    expect(screen.getByText(/dev-only/)).toBeInTheDocument()
  })

  it('offers all four approved materials as a radiogroup', () => {
    render(<DesignLab3D />)
    const group = screen.getByRole('radiogroup', { name: 'Rail material' })
    expect(group).toBeInTheDocument()
    for (const kind of MATERIAL_KINDS) {
      expect(screen.getByTestId(`material-${kind}`)).toBeInTheDocument()
    }
    expect(screen.getByTestId('material-frosted')).toHaveAttribute('aria-checked', 'true')
  })

  it('switches the active material on click', () => {
    render(<DesignLab3D />)
    fireEvent.click(screen.getByTestId('material-obsidian'))
    expect(screen.getByTestId('material-obsidian')).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByTestId('material-frosted')).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByText(/Dark metal bevel/)).toBeInTheDocument()
  })

  it('degrades to a notice when jsdom has no WebGL (no crash)', async () => {
    render(<DesignLab3D />)
    // jsdom cannot create a WebGL context — the component must catch it.
    const fallback = await screen.findByTestId('designlab-webgl-fallback', {}, { timeout: 3000 })
    expect(fallback).toHaveTextContent('WebGL is not available')
  })
})
