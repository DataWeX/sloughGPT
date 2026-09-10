// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { CollectionStatsCard } from './CollectionStatsCard'

afterEach(() => cleanup())

describe('CollectionStatsCard', () => {
  it('renders nothing when null', () => {
    const { container } = render(<CollectionStatsCard stats={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows stats', () => {
    render(<CollectionStatsCard stats={{ pipelines: 5, sources: 3, stores: 2, filters: 8 }} />)
    expect(screen.getByText('Overview')).toBeTruthy()
    expect(screen.getByText('5')).toBeTruthy()
    expect(screen.getByText('3')).toBeTruthy()
    expect(screen.getByText('2')).toBeTruthy()
    expect(screen.getByText('8')).toBeTruthy()
  })

  it('shows labels', () => {
    render(<CollectionStatsCard stats={{ pipelines: 1, sources: 1, stores: 1, filters: 1 }} />)
    expect(screen.getByText('Pipelines')).toBeTruthy()
    expect(screen.getByText('Sources')).toBeTruthy()
    expect(screen.getByText('Stores')).toBeTruthy()
    expect(screen.getByText('Filters')).toBeTruthy()
  })
})
