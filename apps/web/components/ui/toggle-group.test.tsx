/**
 */
import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { ToggleGroup, ToggleGroupItem } from './toggle-group'

afterEach(cleanup)

describe('ToggleGroup', () => {
  it('renders items', () => {
    render(
      <ToggleGroup type="single" value="a">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>
    )
    expect(screen.getByText('A')).toBeInTheDocument()
    expect(screen.getByText('B')).toBeInTheDocument()
  })

  it('renders selected item with aria-checked', () => {
    render(
      <ToggleGroup type="single" value="b">
        <ToggleGroupItem value="a">A</ToggleGroupItem>
        <ToggleGroupItem value="b">B</ToggleGroupItem>
      </ToggleGroup>
    )
    const items = screen.getAllByRole('radio')
    expect(items[1]).toHaveAttribute('aria-checked', 'true')
  })
})
