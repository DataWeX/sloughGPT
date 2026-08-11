// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { FileStatsCard } from './FileStatsCard'
import type { FileEntry } from '@/lib/files-controller'

afterEach(() => { cleanup() })

function makeFile(overrides: Partial<FileEntry> = {}): FileEntry {
  return {
    id: '1',
    filename: 'test.txt',
    size: 1024,
    content_type: 'text/plain',
    ingested: false,
    uploaded_at: new Date().toISOString(),
    ...overrides,
  }
}

describe('FileStatsCard', () => {
  it('returns null for empty files', () => {
    const { container } = render(<FileStatsCard files={[]} />)
    expect(container.querySelector('[data-testid="file-stats"]')).toBeNull()
  })

  it('renders overview with file count and total size', () => {
    render(<FileStatsCard files={[makeFile({ size: 2048 })]} />)
    expect(screen.getAllByTestId('file-stats').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Total').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2.0 KB').length).toBeGreaterThanOrEqual(1)
  })

  it('shows indexed count', () => {
    const { container } = render(<FileStatsCard files={[
      makeFile({ id: '1', ingested: true }),
      makeFile({ id: '2', ingested: false }),
    ]} />)
    expect(screen.getAllByText('Indexed').length).toBeGreaterThanOrEqual(1)
    expect(container.textContent).toContain('1/2')
  })

  it('shows file type groups', () => {
    render(<FileStatsCard files={[
      makeFile({ id: '1', filename: 'a.txt' }),
      makeFile({ id: '2', filename: 'b.py' }),
      makeFile({ id: '3', filename: 'c.json' }),
    ]} />)
    expect(screen.getAllByText('Text').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Code').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Data').length).toBeGreaterThanOrEqual(1)
  })

  it('shows warning for unindexed files', () => {
    render(<FileStatsCard files={[makeFile({ ingested: false })]} />)
    expect(screen.getAllByText(/not indexed/).length).toBeGreaterThanOrEqual(1)
  })

  it('does not show warning when all indexed', () => {
    render(<FileStatsCard files={[makeFile({ ingested: true })]} />)
    expect(screen.queryAllByText(/not indexed/).length).toBe(0)
  })

  it('handles mixed file sizes', () => {
    render(<FileStatsCard files={[
      makeFile({ id: '1', size: 500 }),
      makeFile({ id: '2', size: 2048 }),
      makeFile({ id: '3', size: 5242880 }),
    ]} />)
    expect(screen.getAllByText('Total').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Size/).length).toBeGreaterThanOrEqual(1)
  })

  it('handles unknown file extensions', () => {
    render(<FileStatsCard files={[makeFile({ filename: 'readme' })]} />)
    expect(screen.getAllByText('Other').length).toBeGreaterThanOrEqual(1)
  })
})
