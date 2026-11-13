import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'

vi.mock('@/lib/multimodal-controller', () => ({
  multimodalController: {
    transcribeAudio: vi.fn(),
    getCapabilities: vi.fn().mockResolvedValue({ speech_to_text: true }),
    getTrainingReport: vi.fn().mockResolvedValue({ caption_history: [], vocab_size: 0 }),
    resetModel: vi.fn(),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  IconUpload: (props: any) => <span data-testid="icon-upload" {...props} />,
  IconImage: (props: any) => <span data-testid="icon-image" {...props} />,
  IconX: (props: any) => <span data-testid="icon-x" {...props} />,
  IconMic: (props: any) => <span data-testid="icon-mic" {...props} />,
  AudioWaveform: (props: any) => <span data-testid="audio-waveform" {...props} />,
  IconDocument: (props: any) => <span data-testid="icon-document" {...props} />,
  IconMicFilled: (props: any) => <span data-testid="icon-mic-filled" {...props} />,

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

import { ChatInputAccessories } from './ChatInputAccessories'

describe('ChatInputAccessories', () => {
  const base = {
    onImage: vi.fn(),
    onTranscript: vi.fn(),
    disabled: false,
  }

  it('renders image upload button', () => {
    const { container } = render(<ChatInputAccessories {...base} />)
    const btns = container.querySelectorAll('button[title="Upload image"]')
    expect(btns.length >= 1).toBe(true)
  })

  it('renders audio upload button', () => {
    render(<ChatInputAccessories {...base} />)
    const btns = screen.getAllByRole('button')
    const audioBtn = btns.find(b => b.getAttribute('aria-label') === 'Upload audio')
    expect(audioBtn).toBeInTheDocument()
  })

  it('renders PDFUpload when callbacks provided', () => {
    render(<ChatInputAccessories {...base} onPDFAnalysis={vi.fn()} onPDFError={vi.fn()} />)
    const btns = screen.getAllByRole('button')
    expect(btns.some(b => b.title === 'Upload PDF for analysis')).toBe(true)
  })

  it('does not render PDFUpload when callbacks missing', () => {
    const { container } = render(<ChatInputAccessories {...base} />)
    expect(container.querySelector('[title="Upload PDF for analysis"]')).not.toBeInTheDocument()
  })

  it('passes disabled state', () => {
    const { container } = render(<ChatInputAccessories {...base} disabled />)
    const btns = container.querySelectorAll('button')
    btns.forEach(btn => {
      if (btn.getAttribute('title') !== 'Start voice input' && btn.getAttribute('title') !== 'Stop listening') {
        expect(btn).toBeDisabled()
      }
    })
  })

  it('renders without voice input when callbacks missing', () => {
    const { container } = render(<ChatInputAccessories onImage={vi.fn()} onTranscript={undefined} disabled={false} />)
    const btns = container.querySelectorAll('button')
    const hasVoice = Array.from(btns).some(b =>
      b.getAttribute('title') === 'Start voice input' || b.getAttribute('title') === 'Stop listening'
    )
    expect(hasVoice).toBe(false)
  })

  it('renders all accessory buttons in non-disabled state', () => {
    const { container } = render(<ChatInputAccessories {...base} />)
    const btns = container.querySelectorAll('button')
    const enabled = Array.from(btns).filter(b => !b.disabled)
    expect(enabled.length).toBeGreaterThanOrEqual(2)
  })
})
