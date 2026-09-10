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
