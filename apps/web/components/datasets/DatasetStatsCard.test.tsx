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
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { DatasetStatsCard } from './DatasetStatsCard'

afterEach(() => cleanup())

describe('DatasetStatsCard', () => {
  const defaultProps = {
    totalDatasets: 12,
    totalSize: 5242880,
    totalRows: 150000,
    recentUploads: 3,
  }

  it('renders title', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Dataset Overview')).toBeTruthy()
  })

  it('shows total datasets', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Total Datasets')).toBeTruthy()
    expect(screen.getByText('12')).toBeTruthy()
  })

  it('shows total size formatted', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Total Size')).toBeTruthy()
    expect(screen.getByText('5.0 MB')).toBeTruthy()
  })

  it('shows total rows formatted', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Total Rows')).toBeTruthy()
    expect(screen.getByText('150,000')).toBeTruthy()
  })

  it('shows recent uploads', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Recent Uploads')).toBeTruthy()
    expect(screen.getByText('3')).toBeTruthy()
  })

  it('renders all four stat boxes', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Total Datasets')).toBeTruthy()
    expect(screen.getByText('Total Size')).toBeTruthy()
    expect(screen.getByText('Total Rows')).toBeTruthy()
    expect(screen.getByText('Recent Uploads')).toBeTruthy()
  })
})
