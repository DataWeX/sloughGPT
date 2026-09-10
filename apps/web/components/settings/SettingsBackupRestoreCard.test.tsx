// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  CardFooter: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
  importFile: vi.fn(),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

import { SettingsBackupRestoreCard } from './SettingsBackupRestoreCard'

afterEach(() => cleanup())

describe('SettingsBackupRestoreCard', () => {
  it('renders title and description', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    expect(screen.getByText('Backup & restore')).toBeTruthy()
    expect(screen.getByText('Export your settings to a file, or import from a backup')).toBeTruthy()
  })

  it('renders export button', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    expect(screen.getByText('Export settings')).toBeTruthy()
  })

  it('renders import button', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    expect(screen.getByText('Import settings')).toBeTruthy()
  })

  it('export button is not disabled', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    const btn = screen.getByText('Export settings')
    expect(btn.hasAttribute('disabled')).toBe(false)
  })

  it('import button is not disabled', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} />)
    const btn = screen.getByText('Import settings')
    expect(btn.hasAttribute('disabled')).toBe(false)
  })

  it('renders version badge when provided', () => {
    render(<SettingsBackupRestoreCard settings={{}} onImport={() => {}} version="3.0.0" />)
    expect(screen.getByText(/v3\.0\.0/)).toBeTruthy()
  })
})
