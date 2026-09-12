/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react'
import { ApiKeyCreateCard } from '@/components/api-keys/ApiKeyCreateCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardDescription: ({ children, ...props }: any) => <div data-testid="card-desc" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconPlus: (props: any) => <svg data-testid="icon-plus" {...props} />,
}))

describe('ApiKeyCreateCard', () => {
  const defaultProps = {
    onCreate: vi.fn().mockResolvedValue(undefined),
  }

  it('renders the title', () => {
    render(<ApiKeyCreateCard {...defaultProps} />)
    expect(screen.getByText('Create API Key')).toBeDefined()
  })

  it('renders the name input', () => {
    render(<ApiKeyCreateCard {...defaultProps} />)
    expect(screen.getByTestId('api-key-name')).toBeDefined()
  })

  it('renders the Create button', () => {
    render(<ApiKeyCreateCard {...defaultProps} />)
    expect(screen.getByText('Create')).toBeDefined()
  })

  it('disables Create button when name is empty', () => {
    render(<ApiKeyCreateCard {...defaultProps} />)
    expect(screen.getByText('Create')).toHaveProperty('disabled', true)
  })

  it('updates name input on typing', () => {
    render(<ApiKeyCreateCard {...defaultProps} />)
    fireEvent.change(screen.getByTestId('api-key-name'), { target: { value: 'new-key' } })
    expect(screen.getByTestId('api-key-name')).toHaveValue('new-key')
  })

  it('calls onCreate with the name when Create is clicked', async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined)
    render(<ApiKeyCreateCard onCreate={onCreate} />)
    fireEvent.change(screen.getByTestId('api-key-name'), { target: { value: 'test-key' } })
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => expect(onCreate).toHaveBeenCalledWith('test-key'))
  })

  it('shows Creating while the request is pending', async () => {
    let resolve: (() => void) | undefined
    const onCreate = vi.fn(() => new Promise<void>(r => { resolve = r }))
    render(<ApiKeyCreateCard onCreate={onCreate} />)
    fireEvent.change(screen.getByTestId('api-key-name'), { target: { value: 'key-a' } })
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => expect(screen.getByText('Creating...')).toBeDefined())
    await act(async () => { resolve?.() })
  })
})