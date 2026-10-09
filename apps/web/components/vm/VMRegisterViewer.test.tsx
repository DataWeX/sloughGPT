/// <reference types="vitest" />
// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { VMRegisterViewer } from './VMRegisterViewer'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>{children}</button>
  ),

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

describe('VMRegisterViewer', () => {
  const registers = [
    { name: 'EAX', value: 42, hex: '0x0000002A' },
    { name: 'EBX', value: 0, hex: '0x00000000' },
    { name: 'ECX', value: 7, hex: '0x00000007' },
  ]

  it('renders the title', () => {
    render(<VMRegisterViewer registers={registers} eipHex="0x00001000" />)
    expect(screen.getByText('Registers')).toBeDefined()
  })

  it('renders nothing when registers are empty', () => {
    const { container } = render(<VMRegisterViewer registers={[]} eipHex="0x00001000" />)
    expect(container.firstChild).toBeNull()
  })

  it('renders all register names and hex values', () => {
    render(<VMRegisterViewer registers={registers} eipHex="0x00001000" />)
    expect(screen.getByText('EAX')).toBeDefined()
    expect(screen.getByText('0x0000002A')).toBeDefined()
    expect(screen.getByText('EBX')).toBeDefined()
    expect(screen.getByText('0x00000000')).toBeDefined()
    expect(screen.getByText('ECX')).toBeDefined()
    expect(screen.getByText('0x00000007')).toBeDefined()
  })

  it('renders EIP value', () => {
    render(<VMRegisterViewer registers={registers} eipHex="0x00001000" />)
    expect(screen.getByText('EIP')).toBeDefined()
    expect(screen.getByText('0x00001000')).toBeDefined()
  })

  it('renders Copy button', () => {
    render(<VMRegisterViewer registers={registers} eipHex="0x00001000" />)
    expect(screen.getByText('Copy')).toBeDefined()
  })

  it('calls onCopyAll when Copy is clicked', () => {
    const onCopyAll = vi.fn()
    render(<VMRegisterViewer registers={registers} eipHex="0x00001000" onCopyAll={onCopyAll} />)
    fireEvent.click(screen.getByText('Copy'))
    expect(onCopyAll).toHaveBeenCalledWith(
      'EAX = 0x0000002A\nEBX = 0x00000000\nECX = 0x00000007'
    )
  })

  it('calls onCopyRegister when a register row is clicked', () => {
    const onCopyRegister = vi.fn()
    render(<VMRegisterViewer registers={registers} eipHex="0x00001000" onCopyRegister={onCopyRegister} />)
    fireEvent.click(screen.getByText('EAX').closest('button')!)
    expect(onCopyRegister).toHaveBeenCalledWith('0x0000002A')
  })
})
