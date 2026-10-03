// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/voice-controller', () => ({
  voiceController: {
    getStatus: vi.fn().mockResolvedValue({ server_tts: false, model: 'mock-model', error: null }),
    tts: vi.fn().mockResolvedValue({ duration_ms: 1, backend: 'mock' }),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Textarea: (props: any) => <textarea {...props} />,

  Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
  Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
  ActionCard: ({ title, children }: any) => (
    <div data-testid="action-card">
      <h3>{title}</h3>
      {children}
    </div>
  ),
  Tabs: ({ children }: any) => <div>{children}</div>,
  TabsList: ({ children }: any) => <div>{children}</div>,
  TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  TabsContent: ({ children }: any) => <div>{children}</div>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
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

import { DevVoiceTestCard } from './DevVoiceTestCard'

afterEach(() => cleanup())

// The status effect is fire-and-forget; flush it before assertions so its
// setState can't land after jsdom teardown ("window is not defined"
// unhandled rejections that fail the whole suite run).
async function renderCard() {
  const utils = render(<DevVoiceTestCard />)
  await act(async () => {})
  return utils
}

describe('DevVoiceTestCard', () => {
  it('renders voice status card', async () => {
    await renderCard()
    expect(screen.getByText('Voice Status')).toBeTruthy()
  })

  it('renders quick test card', async () => {
    await renderCard()
    expect(screen.getByText('Quick Test')).toBeTruthy()
  })

  it('renders TTS input', async () => {
    await renderCard()
    expect(screen.getByLabelText('Text to speech input')).toBeTruthy()
  })

  it('renders speak button', async () => {
    await renderCard()
    expect(screen.getByRole('button', { name: /Speak/ })).toBeTruthy()
  })

  it('speak button disabled when empty', async () => {
    await renderCard()
    expect(screen.getByRole('button', { name: /Speak/ }).getAttribute('disabled')).not.toBeNull()
  })
})
