import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/format-bytes', () => ({
  formatDuration: (ms: number) => {
    const s = Math.floor(ms / 1000)
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
  },
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  IconPlay: (p: any) => <svg {...p} data-testid="icon-play" />,
  IconStop: (p: any) => <svg {...p} data-testid="icon-stop" />,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
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

import { AudioPlayer } from './AudioPlayer'

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(cleanup)

describe('AudioPlayer', () => {
  it('renders without crashing', () => {
    render(<AudioPlayer src="/audio.mp3" />)
    expect(screen.getByRole('button', { name: /Play/i })).toBeDefined()
  })

  it('renders with custom className', () => {
    render(<AudioPlayer src="/audio.mp3" className="custom-class" />)
    expect(screen.getByRole('button', { name: /Play/i })).toBeDefined()
  })

  it('shows play button with play label', () => {
    render(<AudioPlayer src="/audio.mp3" />)
    expect(screen.getByLabelText('Play')).toBeDefined()
  })

  it('toggles play/pause on click', () => {
    render(<AudioPlayer src="/audio.mp3" />)
    const btn = screen.getByRole('button', { name: /Play/i })
    fireEvent.click(btn)
    expect(screen.getByLabelText('Pause')).toBeDefined()
    fireEvent.click(btn)
    expect(screen.getByLabelText('Play')).toBeDefined()
  })

  it('displays formatted duration text', () => {
    render(<AudioPlayer src="/audio.mp3" durationMs={60000} />)
    expect(screen.getByText('1:00')).toBeDefined()
  })

  it('displays audio element with src', () => {
    render(<AudioPlayer src="/test-audio.wav" />)
    const audio = document.querySelector('audio')
    expect(audio).toBeDefined()
    expect(audio?.getAttribute('src')).toBe('/test-audio.wav')
  })

  it('renders progress bar structure', () => {
    render(<AudioPlayer src="/audio.mp3" />)
    expect(document.querySelector('[class*="bg-muted"]')).toBeDefined()
    expect(document.querySelector('[class*="bg-primary"]')).toBeDefined()
  })

  it('displays time labels', () => {
    render(<AudioPlayer src="/audio.mp3" durationMs={120000} />)
    const timeLabels = screen.getAllByText(/\d+:\d{2}/)
    expect(timeLabels.length).toBeGreaterThanOrEqual(2)
  })
})
