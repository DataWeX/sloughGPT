/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { VectorInitCard } from './VectorInitCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
}))

afterEach(() => cleanup())

describe('VectorInitCard', () => {
  it('renders the card title', () => {
    render(<VectorInitCard provider="in_memory" loading={false} onInit={vi.fn()} />)
    expect(screen.getByText('Initialize')).toBeDefined()
  })

  it('renders In Memory button', () => {
    render(<VectorInitCard provider="in_memory" loading={false} onInit={vi.fn()} />)
    expect(screen.getByText('In Memory')).toBeDefined()
  })

  it('renders ChromaDB button', () => {
    render(<VectorInitCard provider="in_memory" loading={false} onInit={vi.fn()} />)
    expect(screen.getByText('ChromaDB')).toBeDefined()
  })

  it('calls onInit with in_memory when clicked', async () => {
    const onInit = vi.fn()
    render(<VectorInitCard provider="in_memory" loading={false} onInit={onInit} />)
    await userEvent.click(screen.getByText('In Memory'))
    expect(onInit).toHaveBeenCalledWith('in_memory')
  })

  it('calls onInit with chromadb when clicked', async () => {
    const onInit = vi.fn()
    render(<VectorInitCard provider="in_memory" loading={false} onInit={onInit} />)
    await userEvent.click(screen.getByText('ChromaDB'))
    expect(onInit).toHaveBeenCalledWith('chromadb')
  })
})
