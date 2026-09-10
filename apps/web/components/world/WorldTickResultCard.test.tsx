/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { WorldTickResultCard } from './WorldTickResultCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

describe('WorldTickResultCard', () => {
  it('renders nothing when both results are null', () => {
    const { container } = render(<WorldTickResultCard tickResult={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders tick result title', () => {
    render(<WorldTickResultCard tickResult={{ tick: 1, babies: 3 }} />)
    expect(screen.getByText('Tick Result')).toBeDefined()
  })

  it('renders tick number', () => {
    render(<WorldTickResultCard tickResult={{ tick: 5, babies: 0 }} />)
    expect(screen.getByText('5')).toBeDefined()
  })

  it('renders babies count', () => {
    render(<WorldTickResultCard tickResult={{ tick: 1, babies: 12 }} />)
    expect(screen.getByText('12')).toBeDefined()
  })

  it('renders neural processing title when neuralResult provided', () => {
    render(
      <WorldTickResultCard
        tickResult={{ tick: 1, babies: 0 }}
        neuralResult={{ embedding_shape: [128, 64] }}
      />,
    )
    expect(screen.getByText('Neural Processing')).toBeDefined()
  })

  it('renders embedding shape', () => {
    render(
      <WorldTickResultCard
        tickResult={{ tick: 1, babies: 0 }}
        neuralResult={{ embedding_shape: [128, 64] }}
      />,
    )
    expect(screen.getByText('128,64')).toBeDefined()
  })

  it('renders N/A when no embedding shape', () => {
    render(
      <WorldTickResultCard
        tickResult={{ tick: 1, babies: 0 }}
        neuralResult={{ descriptor: { foo: 'bar' } }}
      />,
    )
    expect(screen.getByText('N/A')).toBeDefined()
  })

  it('renders descriptor JSON', () => {
    render(
      <WorldTickResultCard
        tickResult={{ tick: 1, babies: 0 }}
        neuralResult={{ descriptor: { key: 'value' } }}
      />,
    )
    expect(screen.getByText(/"key": "value"/)).toBeDefined()
  })

  it('renders neural card even without tick result', () => {
    render(
      <WorldTickResultCard
        tickResult={null}
        neuralResult={{ embedding_shape: [10] }}
      />,
    )
    expect(screen.getByText('Neural Processing')).toBeDefined()
  })
})
