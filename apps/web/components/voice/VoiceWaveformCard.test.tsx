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
  addEventListener: vi.fn((event: string, cb: Function, opts?: any) => {
    if (event === 'canplaythrough') setTimeout(cb, 0)
    if (event === 'error') { /* noop */ }
  }),
  load: vi.fn(),
})))

vi.stubGlobal('AudioContext', vi.fn().mockImplementation(() => ({
  createMediaElementSource: vi.fn().mockReturnValue({ connect: vi.fn() }),
  createAnalyser: vi.fn().mockReturnValue({
    fftSize: 256,
    frequencyBinCount: 128,
    getFloatTimeDomainData: vi.fn(),
    getFloatFrequencyData: vi.fn(),
  }),
  destination: {},
  close: vi.fn(),
})))

import { VoiceWaveformCard } from './VoiceWaveformCard'

beforeEach(() => { localStorage.clear() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('VoiceWaveformCard', () => {
  it('renders the component', () => {
    render(<VoiceWaveformCard />)
    expect(screen.getByTestId('voice-waveform')).toBeTruthy()
  })

  it('shows empty state message when no recordings', async () => {
    render(<VoiceWaveformCard />)
    await waitFor(() => {
      expect(screen.getByText('No recordings yet. Record something first.')).toBeTruthy()
    })
  })

  it('renders select with recordings', async () => {
    localStorage.setItem('sloughgpt-voice-recordings', JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 5000, label: 'Voice Note', timestamp: 1 },
    ]))
    render(<VoiceWaveformCard />)
    await waitFor(() => {
      expect(screen.getByText('Voice Note (5.0s)')).toBeTruthy()
    })
  })

  it('shows title', () => {
    render(<VoiceWaveformCard />)
    expect(screen.getByText('Waveform Analysis')).toBeTruthy()
  })
})
