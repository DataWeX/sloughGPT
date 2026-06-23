// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

import { CustomDropdown } from './CustomDropdown'

const defaultItems = [
  { label: 'Option 1', onClick: vi.fn() },
  { label: 'Option 2', onClick: vi.fn() },
  { label: 'Delete', onClick: vi.fn(), destructive: true },
  { label: 'Separated', onClick: vi.fn(), separator: true },
  { label: 'Custom', custom: <span data-testid="custom-item">Custom</span> },
]

describe('CustomDropdown', () => {
  afterEach(cleanup)

  it('renders trigger', () => {
    render(<CustomDropdown trigger={<button>Menu</button>} items={[]} />)
    expect(screen.getByText('Menu')).toBeDefined()
  })

  it('shows menu on trigger click', () => {
    render(<CustomDropdown trigger={<button>Menu</button>} items={defaultItems} />)
    fireEvent.click(screen.getByText('Menu'))
    expect(screen.getByText('Option 1')).toBeDefined()
    expect(screen.getByText('Option 2')).toBeDefined()
  })

  it('calls onClick and closes on item click', () => {
    const onClick = vi.fn()
    render(<CustomDropdown trigger={<button>Menu</button>} items={[{ label: 'Option', onClick }]} />)
    fireEvent.click(screen.getByText('Menu'))
    fireEvent.click(screen.getByText('Option'))
    expect(onClick).toHaveBeenCalled()
    expect(screen.queryByText('Option')).toBeNull()
  })

  it('closes on outside click', () => {
    render(<CustomDropdown trigger={<button>Menu</button>} items={defaultItems} />)
    fireEvent.click(screen.getByText('Menu'))
    expect(screen.getByText('Option 1')).toBeDefined()
    fireEvent.mouseDown(document.body)
    expect(screen.queryByText('Option 1')).toBeNull()
  })

  it('renders custom items', () => {
    render(<CustomDropdown trigger={<button>Menu</button>} items={[{ custom: <span data-testid="custom-item">Custom</span> }]} />)
    fireEvent.click(screen.getByText('Menu'))
    expect(screen.getByTestId('custom-item')).toBeDefined()
  })
})
