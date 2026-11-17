// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { DatasetChip, DatasetTooltipContent, formatSize } from './dataset-chip'
import { ds } from './__test-helper'

vi.mock('@sloughgpt/strui', () => ({
  Badge: ({ children }: any) => <span data-testid="badge">{children}</span>,
  Tooltip: ({ children }: any) => <div>{children}</div>,
  TooltipTrigger: ({ children }: any) => <div>{children}</div>,
  TooltipContent: ({ children }: any) => <div>{children}</div>,
}))

afterEach(cleanup)

describe('DatasetTooltipContent — empty import honesty', () => {
  // card 20261115_081635_ux…: the 221 B Gutenberg metadata stub must surface
  // as empty, never as a ready dataset with a healthy size.
  const stub = ds({
    name: "J.R.R. Tolkien's The hobbit",
    size: 0,
    samples: 0,
    has_corpus: false,
  })

  it('renders Empty + warning for a metadata-only stub', () => {
    render(<DatasetTooltipContent dataset={stub} />)
    expect(screen.getByText('Empty')).toBeDefined()
    expect(screen.getByText(/no training text retrieved/i)).toBeDefined()
    expect(screen.queryByText(/221 B/)).toBeNull()
  })

  it('renders normal size for a healthy corpus', () => {
    render(<DatasetTooltipContent dataset={ds({ name: 'shakespeare', size: 2048, samples: 42 })} />)
    expect(screen.getByText('2.0 KB')).toBeDefined()
    expect(screen.getByText('42')).toBeDefined()
    expect(screen.queryByText(/no training text retrieved/i)).toBeNull()
  })

  it('treats size-less, sample-less entries without has_corpus as empty', () => {
    render(<DatasetTooltipContent dataset={ds({ name: 'ghost', size: 0 })} />)
    expect(screen.getByText('Empty')).toBeDefined()
  })

  it('never marks VLM entries empty (images, not text)', () => {
    render(
      <DatasetTooltipContent
        dataset={ds({ name: 'faces', type: 'vlm', size: 0, samples: 0, has_corpus: false })}
      />,
    )
    expect(screen.queryByText('Empty')).toBeNull()
  })
})

describe('DatasetChip', () => {
  it('shows a warning dot on empty datasets — via testid, never a native title', () => {
    const { container } = render(
      <DatasetChip
        dataset={ds({ name: 'stub', size: 0, has_corpus: false })}
        selected={false}
        onSelect={vi.fn()}
      />,
    )
    expect(screen.getByTestId('dataset-empty-dot')).toBeDefined()
    // Regression: a native title fired alongside the custom tooltip on the
    // same hover (double tooltip, raw id in the browser box).
    expect(container.querySelector('[title]')).toBeNull()
  })

  it('shows sample count instead of the dot when healthy', () => {
    const { container } = render(
      <DatasetChip
        dataset={ds({ name: 'ok', size: 4096, samples: 7 })}
        selected={false}
        onSelect={vi.fn()}
      />,
    )
    expect(screen.queryByTestId('dataset-empty-dot')).toBeNull()
    expect(container.querySelector('[title]')).toBeNull()
    // Chip badge + tooltip row both render "7" in this flattened mock.
    expect(screen.getAllByText('7').length).toBeGreaterThan(0)
  })
})

describe('formatSize', () => {
  it('formats byte ranges', () => {
    expect(formatSize(0)).toBe('Unknown size')
    expect(formatSize(512)).toBe('512 B')
    expect(formatSize(2048)).toBe('2.0 KB')
    expect(formatSize(null)).toBe('Unknown size')
  })
})
