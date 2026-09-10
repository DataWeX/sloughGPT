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
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,
}))

vi.mock('lucide-react', () => ({
  Search: () => null,
  Trash2: () => null,
  ArrowUpDown: () => null,
}))

import { DatasetListCard } from './DatasetListCard'

afterEach(() => cleanup())

const mockDatasets = [
  { id: '1', name: 'train.jsonl', size: 1048576, row_count: 5000, updated_at: '2025-01-10', tags: ['train'] },
  { id: '2', name: 'eval.jsonl', size: 262144, row_count: 1000, updated_at: '2025-01-12' },
]

describe('DatasetListCard', () => {
  const defaultProps = {
    datasets: mockDatasets,
    loading: false,
    search: '',
    sortBy: 'date' as const,
    onSearchChange: vi.fn(),
    onSortChange: vi.fn(),
    onSelect: vi.fn(),
    onDelete: vi.fn(),
  }

  it('renders title', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByText('Datasets')).toBeTruthy()
  })

  it('renders search input', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByTestId('dataset-search')).toBeTruthy()
  })

  it('renders sort buttons', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByText('Date')).toBeTruthy()
    expect(screen.getByText('Size')).toBeTruthy()
    expect(screen.getByText('Name')).toBeTruthy()
  })

  it('renders dataset entries when provided', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByText('train.jsonl')).toBeTruthy()
    expect(screen.getByText('eval.jsonl')).toBeTruthy()
    expect(screen.getByText('5,000 rows')).toBeTruthy()
    expect(screen.getByText('1,000 rows')).toBeTruthy()
  })

  it('shows loading placeholders', () => {
    render(<DatasetListCard {...defaultProps} loading={true} datasets={[]} />)
    expect(screen.queryByText('train.jsonl')).toBeNull()
  })

  it('shows empty message', () => {
    render(<DatasetListCard {...defaultProps} datasets={[]} />)
    expect(screen.getByText('No datasets found.')).toBeTruthy()
  })

  it('renders tags when present', () => {
    render(<DatasetListCard {...defaultProps} />)
    expect(screen.getByText('train')).toBeTruthy()
  })
})
