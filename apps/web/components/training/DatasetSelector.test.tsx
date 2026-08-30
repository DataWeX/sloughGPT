// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { datasetLabel, DatasetSelector } from './DatasetSelector'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import { ds, makeDatasets } from './__test-helper'

vi.mock('@/components/DatasetImportDialog', () => ({
  DatasetImportDialog: () => <div data-testid="import-modal" />,
}))

vi.mock('@sloughgpt/strui', () => ({
  Select: ({ children, ...props }: any) => <select data-testid="select" {...props}>{children}</select>,
  SelectTrigger: ({ children }: any) => <div>{children}</div>,
  SelectValue: () => <span>Select a dataset...</span>,
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children, value }: any) => <option value={value}>{children}</option>,
  Button: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
}))

const datasets: UseTrainingDatasetsReturn = makeDatasets()

describe('datasetLabel', () => {
  it('returns name only for basic dataset', () => {
    expect(datasetLabel(ds({ name: 'shakespeare' }))).toContain('shakespeare')
  })

  it('includes sample count', () => {
    expect(datasetLabel(ds({ name: 'ds', samples: 100 }))).toContain('100 samples')
  })

  it('includes source when present', () => {
    expect(datasetLabel(ds({ name: 'ds', source: 'github' }))).toContain('github')
  })

  it('includes size when present', () => {
    expect(datasetLabel(ds({ name: 'ds', size: 2048 }))).toContain('2.0 KB')
  })
})

describe('DatasetSelector', () => {
  afterEach(cleanup)

  it('shows empty state when no datasets', () => {
    render(<DatasetSelector datasets={datasets} value="" onChange={vi.fn()} />)
    expect(screen.getByText(/No datasets/)).toBeDefined()
  })

  it('shows import button in empty state', () => {
    render(<DatasetSelector datasets={datasets} value="" onChange={vi.fn()} />)
    expect(screen.getByText(/Import/)).toBeDefined()
  })

  it('shows selector when datasets exist', () => {
    const withData = { ...datasets, datasets: [ds({ name: 'shakespeare' })] }
    render(<DatasetSelector datasets={withData} value="" onChange={vi.fn()} />)
    expect(screen.getByTestId('select')).toBeDefined()
  })

  it('shows import button when showImport is true', () => {
    const withData = { ...datasets, datasets: [ds({ name: 'shakespeare' })] }
    render(<DatasetSelector datasets={withData} value="" onChange={vi.fn()} showImport />)
    expect(screen.getByText(/Import/)).toBeDefined()
  })
})
