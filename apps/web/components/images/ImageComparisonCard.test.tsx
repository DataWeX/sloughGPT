// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { ImageComparisonCard } from './ImageComparisonCard'

afterEach(() => cleanup())

const mockImages = [
  { id: 'img-1', label: 'Original', src: 'data:image/png;base64,orig' },
  { id: 'img-2', label: 'Enhanced', src: 'data:image/png;base64,enh' },
]

describe('ImageComparisonCard', () => {
  it('renders empty state', () => {
    render(<ImageComparisonCard />)
    expect(screen.getByText('Image Comparison')).toBeTruthy()
    expect(screen.getByText('Select two images to compare')).toBeTruthy()
  })

  it('shows select dropdowns', () => {
    render(<ImageComparisonCard images={mockImages} />)
    const selects = screen.getAllByRole('combobox')
    expect(selects.length).toBe(2)
  })

  it('renders with images prop', () => {
    render(<ImageComparisonCard images={mockImages} />)
    expect(screen.getByText('Image Comparison')).toBeTruthy()
    expect(screen.getByText('Select image A')).toBeTruthy()
    expect(screen.getByText('Select image B')).toBeTruthy()
  })

  it('switches comparison mode', () => {
    render(<ImageComparisonCard images={mockImages} />)
    const sliderBtn = screen.getAllByRole('button').find(b => b.textContent === 'Slider')
    fireEvent.click(sliderBtn!)
    expect(sliderBtn).toBeTruthy()
    const overlayBtn = screen.getAllByRole('button').find(b => b.textContent === 'Overlay')
    fireEvent.click(overlayBtn!)
    expect(overlayBtn).toBeTruthy()
  })

  it('selects images from dropdown', () => {
    render(<ImageComparisonCard images={mockImages} />)
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'img-1' } })
    fireEvent.change(selects[1], { target: { value: 'img-2' } })
    expect(screen.getAllByRole('img').length).toBeGreaterThanOrEqual(2)
  })
})
