/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { TokenTreeVocabTable, type VocabEntry } from './TokenTreeVocabTable'

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

const entries: VocabEntry[] = [
  { id: 0, token: 'hello', freq: 42, is_special: false, is_merged: true },
  { id: 1, token: 'world', freq: 17, is_special: false, is_merged: false },
  { id: 2, token: '[PAD]', freq: 0, is_special: true, is_merged: false },
]

describe('TokenTreeVocabTable', () => {
  it('renders the vocabulary title with count', () => {
    render(<TokenTreeVocabTable entries={[]} total={100} offset={0} onPageChange={vi.fn()} />)
    expect(screen.getByText('Vocabulary (100 tokens)')).toBeDefined()
  })

  it('renders token text in the table', () => {
    render(<TokenTreeVocabTable entries={entries} total={3} offset={0} onPageChange={vi.fn()} />)
    expect(screen.getByText('hello')).toBeDefined()
    expect(screen.getByText('world')).toBeDefined()
  })

  it('shows special badge for special tokens', () => {
    render(<TokenTreeVocabTable entries={entries} total={3} offset={0} onPageChange={vi.fn()} />)
    expect(screen.getByText('special')).toBeDefined()
  })

  it('shows merged badge for merged tokens', () => {
    render(<TokenTreeVocabTable entries={entries} total={3} offset={0} onPageChange={vi.fn()} />)
    expect(screen.getByText('merged')).toBeDefined()
  })

  it('calls onPageChange when Next is clicked', async () => {
    const onPageChange = vi.fn()
    render(<TokenTreeVocabTable entries={entries} total={100} offset={0} onPageChange={onPageChange} />)
    await userEvent.click(screen.getByText('Next'))
    expect(onPageChange).toHaveBeenCalledWith(50)
  })
})
