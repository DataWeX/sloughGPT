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
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { AuditTrailFilterCard } from './AuditTrailFilterCard'

afterEach(() => cleanup())

describe('AuditTrailFilterCard', () => {
  it('renders filter controls', () => {
    render(<AuditTrailFilterCard />)
    expect(screen.getByTestId('audit-search')).toBeTruthy()
    expect(screen.getByTestId('audit-type-filter')).toBeTruthy()
    expect(screen.getByTestId('audit-date-from')).toBeTruthy()
    expect(screen.getByTestId('audit-date-to')).toBeTruthy()
  })

  it('shows type options', () => {
    render(<AuditTrailFilterCard types={['training', 'audit']} />)
    const select = screen.getByTestId('audit-type-filter')
    expect(select.querySelectorAll('option').length).toBe(3)
  })

  it('calls onFilterChange on text input', () => {
    const onFilterChange = vi.fn()
    render(<AuditTrailFilterCard onFilterChange={onFilterChange} />)
    fireEvent.change(screen.getByTestId('audit-search'), { target: { value: 'login' } })
    expect(onFilterChange).toHaveBeenCalledWith(expect.objectContaining({ text: 'login' }))
  })

  it('calls onFilterChange on type change', () => {
    const onFilterChange = vi.fn()
    render(<AuditTrailFilterCard types={['training']} onFilterChange={onFilterChange} />)
    fireEvent.change(screen.getByTestId('audit-type-filter'), { target: { value: 'training' } })
    expect(onFilterChange).toHaveBeenCalledWith(expect.objectContaining({ type: 'training' }))
  })

  it('shows clear button when filters active', () => {
    render(<AuditTrailFilterCard onFilterChange={vi.fn()} />)
    fireEvent.change(screen.getByTestId('audit-date-from'), { target: { value: '2026-01-01' } })
    expect(screen.getByText('Clear')).toBeTruthy()
  })

  it('clears all filters', () => {
    const onFilterChange = vi.fn()
    render(<AuditTrailFilterCard onFilterChange={onFilterChange} />)
    fireEvent.change(screen.getByTestId('audit-date-from'), { target: { value: '2026-01-01' } })
    fireEvent.click(screen.getByText('Clear'))
    expect(onFilterChange).toHaveBeenCalledWith({ text: '', type: 'all', dateFrom: '', dateTo: '' })
  })

  it('shows export button', () => {
    render(<AuditTrailFilterCard onExport={vi.fn()} />)
    expect(screen.getByText('Export CSV')).toBeTruthy()
  })
})
