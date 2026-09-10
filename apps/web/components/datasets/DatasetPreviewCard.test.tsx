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
}))

import { DatasetPreviewCard } from './DatasetPreviewCard'

afterEach(() => cleanup())

describe('DatasetPreviewCard', () => {
  const defaultProps = {
    datasetName: 'train.jsonl',
    preview: null,
    loading: false,
    search: '',
    onSearchChange: vi.fn(),
  }

  it('renders title with dataset name', () => {
    render(<DatasetPreviewCard {...defaultProps} />)
    expect(screen.getByText('Preview: train.jsonl')).toBeTruthy()
  })

  it('renders search input', () => {
    render(<DatasetPreviewCard {...defaultProps} />)
    expect(screen.getByTestId('preview-search')).toBeTruthy()
  })

  it('shows no data message when preview is null', () => {
    render(<DatasetPreviewCard {...defaultProps} />)
    expect(screen.getByText('No preview data available.')).toBeTruthy()
  })

  it('renders table with column headers', () => {
    const preview = { columns: ['id', 'text', 'label'], rows: [['1', 'hello', 'pos']] }
    render(<DatasetPreviewCard {...defaultProps} preview={preview} />)
    expect(screen.getByText('id')).toBeTruthy()
    expect(screen.getByText('text')).toBeTruthy()
    expect(screen.getByText('label')).toBeTruthy()
  })

  it('renders table rows', () => {
    const preview = {
      columns: ['name', 'value'],
      rows: [['alpha', '100'], ['beta', '200']],
    }
    render(<DatasetPreviewCard {...defaultProps} preview={preview} />)
    expect(screen.getByText('alpha')).toBeTruthy()
    expect(screen.getByText('100')).toBeTruthy()
    expect(screen.getByText('beta')).toBeTruthy()
    expect(screen.getByText('200')).toBeTruthy()
  })

  it('shows loading state', () => {
    render(<DatasetPreviewCard {...defaultProps} loading={true} />)
    expect(screen.queryByText('No preview data available.')).toBeNull()
  })

  it('renders table element', () => {
    const preview = { columns: ['col'], rows: [['val']] }
    render(<DatasetPreviewCard {...defaultProps} preview={preview} />)
    expect(screen.getByTestId('preview-table')).toBeTruthy()
  })
})
