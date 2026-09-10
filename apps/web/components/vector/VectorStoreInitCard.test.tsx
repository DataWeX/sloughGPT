import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { VectorStoreInitCard } from './VectorStoreInitCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, variant, ...props }: any) => (
    <button data-testid={`btn-${variant || 'default'}`} onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
}))

const defaultProps = {
  provider: 'in_memory',
  initializing: false,
  onInit: vi.fn(),
}

describe('VectorStoreInitCard', () => {
  it('renders the card title', () => {
    render(<VectorStoreInitCard {...defaultProps} />)
    expect(screen.getByText('Initialize')).toBeDefined()
  })

  it('renders the description text', () => {
    render(<VectorStoreInitCard {...defaultProps} />)
    expect(screen.getByText(/Choose a vector store backend/)).toBeDefined()
  })

  it('renders In Memory and ChromaDB buttons', () => {
    render(<VectorStoreInitCard {...defaultProps} />)
    expect(screen.getByText('In Memory')).toBeDefined()
    expect(screen.getByText('ChromaDB')).toBeDefined()
  })

  it('calls onInit with in_memory when clicked', async () => {
    const onInit = vi.fn()
    render(<VectorStoreInitCard {...defaultProps} onInit={onInit} />)
    await userEvent.click(screen.getByText('In Memory'))
    expect(onInit).toHaveBeenCalledWith('in_memory')
  })

  it('calls onInit with chromadb when clicked', async () => {
    const onInit = vi.fn()
    render(<VectorStoreInitCard {...defaultProps} onInit={onInit} />)
    await userEvent.click(screen.getByText('ChromaDB'))
    expect(onInit).toHaveBeenCalledWith('chromadb')
  })

  it('disables buttons when initializing', () => {
    render(<VectorStoreInitCard {...defaultProps} initializing={true} />)
    expect((screen.getByText('In Memory') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('ChromaDB') as HTMLButtonElement).disabled).toBe(true)
  })
})
