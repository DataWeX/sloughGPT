// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { ExportProgressCard, startExportJob, updateExportJob } from './ExportProgressCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('ExportProgressCard', () => {
  it('renders nothing when no jobs', () => {
    const { container } = render(<ExportProgressCard />)
    expect(container.firstChild).toBeNull()
  })

  it('shows running jobs', () => {
    const id = startExportJob('Model Export')
    updateExportJob(id, { progress: 50, message: 'Processing...' })
    render(<ExportProgressCard />)
    expect(screen.getByText('Export Progress')).toBeTruthy()
    expect(screen.getByText('Model Export')).toBeTruthy()
    expect(screen.getByText('50%')).toBeTruthy()
    expect(screen.getByText('Processing...')).toBeTruthy()
  })

  it('shows completed jobs', () => {
    const id = startExportJob('Training Data')
    updateExportJob(id, { status: 'completed', progress: 100 })
    render(<ExportProgressCard />)
    expect(screen.getByText('completed')).toBeTruthy()
  })

  it('shows failed jobs with error', () => {
    const id = startExportJob('Checkpoint')
    updateExportJob(id, { status: 'failed', error: 'Disk full' })
    render(<ExportProgressCard />)
    expect(screen.getByText('failed')).toBeTruthy()
    expect(screen.getByText('Disk full')).toBeTruthy()
  })

  it('shows active count badge', () => {
    startExportJob('Job 1')
    startExportJob('Job 2')
    render(<ExportProgressCard />)
    expect(screen.getByText('2 active')).toBeTruthy()
  })

  it('clears completed jobs', () => {
    const id = startExportJob('Done Job')
    updateExportJob(id, { status: 'completed', progress: 100 })
    render(<ExportProgressCard />)
    fireEvent.click(screen.getByText('Clear done'))
    expect(screen.queryByText('Done Job')).toBeNull()
  })
})
