// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn((key: string) => {
      const raw = localStorage.getItem(key)
      return Promise.resolve(raw ? JSON.parse(raw) : undefined)
    }),
    setKV: vi.fn().mockResolvedValue(undefined),
    deleteKV: vi.fn().mockResolvedValue(undefined),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Select: ({ children, value, onValueChange, ...props }: any) => (
    <select data-testid={props['aria-label']} value={value} onChange={(e) => onValueChange(e.target.value)}>{children}</select>
  ),
  SelectTrigger: ({ children }: any) => <div>{children}</div>,
  SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children, value }: any) => <option value={value}>{children}</option>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

vi.stubGlobal('Audio', vi.fn().mockImplementation(() => ({
  play: vi.fn(),
  addEventListener: vi.fn(),
  load: vi.fn(),
})))

vi.stubGlobal('AudioContext', vi.fn().mockImplementation(() => ({
  createMediaElementSource: vi.fn().mockReturnValue({ connect: vi.fn() }),
  createAnalyser: vi.fn().mockReturnValue({
    fftSize: 128,
    frequencyBinCount: 64,
    getFloatTimeDomainData: vi.fn(),
  }),
  destination: {},
  close: vi.fn(),
})))

import { VoiceComparisonCard } from './VoiceComparisonCard'

const STORAGE_KEY = 'sloughgpt-voice-recordings'

beforeEach(() => { localStorage.clear() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('VoiceComparisonCard', () => {
  it('shows message when no recordings exist', async () => {
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Record at least 2 clips to compare.')).toBeTruthy()
    })
  })

  it('shows message when only 1 recording exists', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 3000, label: 'Clip A', timestamp: 1 },
    ]))
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Record at least 1 more clip to compare.')).toBeTruthy()
    })
  })

  it('renders comparison form with 2+ recordings', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 3000, label: 'Clip A', timestamp: 1 },
      { id: 'r2', url: 'blob:b', duration: 4000, label: 'Clip B', timestamp: 2 },
    ]))
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Voice Comparison')).toBeTruthy()
      expect(screen.getByText('Compare')).toBeTruthy()
    })
  })

  it('Compare button is disabled without selections', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 3000, label: 'Clip A', timestamp: 1 },
      { id: 'r2', url: 'blob:b', duration: 4000, label: 'Clip B', timestamp: 2 },
    ]))
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Compare')).toBeTruthy()
    })
    expect(screen.getByText('Compare')).toHaveProperty('disabled', true)
  })

  it('displays recording options in select', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 3000, label: 'Morning Voice', timestamp: 1 },
      { id: 'r2', url: 'blob:b', duration: 4000, label: 'Evening Voice', timestamp: 2 },
    ]))
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Voice Comparison')).toBeTruthy()
      const options = document.querySelectorAll('option')
      expect(options.length).toBeGreaterThanOrEqual(3)
    })
  })
})
