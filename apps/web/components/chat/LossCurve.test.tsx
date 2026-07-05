import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'

import { LossCurve } from './LossCurve'

describe('LossCurve', () => {
  it('renders SVG with aria-label', () => {
    render(<LossCurve data={[{ step: 0, loss: 1 }, { step: 1, loss: 0.5 }]} />)
    expect(screen.getByLabelText('Training loss over steps')).toBeInTheDocument()
  })

  it('renders polyline when 2+ data points', () => {
    const { container } = render(<LossCurve data={[{ step: 0, loss: 1 }, { step: 1, loss: 0.5 }]} />)
    expect(container.querySelector('polyline')).toBeInTheDocument()
  })

  it('renders fill path when 2+ data points', () => {
    const { container } = render(<LossCurve data={[{ step: 0, loss: 1 }, { step: 1, loss: 0.5 }]} />)
    expect(container.querySelector('path')).toBeInTheDocument()
  })

  it('renders endpoint dot', () => {
    const { container } = render(<LossCurve data={[{ step: 0, loss: 1 }, { step: 1, loss: 0.5 }]} />)
    expect(container.querySelector('circle')).toBeInTheDocument()
  })

  it('renders step labels with min/max', () => {
    const { container } = render(<LossCurve data={[{ step: 0, loss: 1 }, { step: 1, loss: 0.5 }]} />)
    const labels = container.querySelector('.flex.justify-between')
    expect(labels).toHaveTextContent(/step 0/)
    expect(labels).toHaveTextContent(/step 1/)
    expect(labels).toHaveTextContent(/1.00/)
  })

  it('does not crash with single data point', () => {
    const { container } = render(<LossCurve data={[{ step: 0, loss: 1 }]} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
    expect(container.querySelector('polyline')).not.toBeInTheDocument()
  })

  it('does not crash with empty data', () => {
    const { container } = render(<LossCurve data={[]} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })
})
