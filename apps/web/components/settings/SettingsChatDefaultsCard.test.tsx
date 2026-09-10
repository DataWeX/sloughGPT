// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
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
  Slider: ({ value, onValueChange, ...props }: any) => (
    <input type="range" value={value?.[0] ?? 0} onChange={(e) => onValueChange?.([Number(e.target.value)])} {...props} />
  ),
  Switch: ({ checked, onCheckedChange, ...props }: any) => (
    <input type="checkbox" checked={checked} onChange={(e) => onCheckedChange?.(e.target.checked)} {...props} />
  ),
}))

import { SettingsChatDefaultsCard } from './SettingsChatDefaultsCard'

afterEach(() => cleanup())

const defaultProps = {
  temperature: 0.7,
  maxTokens: 512,
  topP: 0.9,
  topK: 50,
  streaming: true,
  collapsibleMessageLength: 200,
  onTemperatureChange: vi.fn(),
  onMaxTokensChange: vi.fn(),
  onTopPChange: vi.fn(),
  onTopKChange: vi.fn(),
  onStreamingChange: vi.fn(),
  onCollapsibleMessageLengthChange: vi.fn(),
  onReset: vi.fn(),
  version: '3.0.0',
}

describe('SettingsChatDefaultsCard', () => {
  it('renders title and description', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    expect(screen.getByText('Chat defaults')).toBeTruthy()
    expect(screen.getByText('Default model and generation settings')).toBeTruthy()
  })

  it('displays temperature slider value', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    expect(screen.getByText('Temperature')).toBeTruthy()
    expect(screen.getByText('0.7')).toBeTruthy()
  })

  it('displays streaming toggle', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    expect(screen.getByText('Streaming')).toBeTruthy()
    expect(screen.getByText('Show tokens as they are generated')).toBeTruthy()
  })

  it('calls onStreamingChange when streaming toggled', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    const toggle = screen.getByRole('checkbox', { name: /toggle streaming/i })
    fireEvent.click(toggle)
    expect(defaultProps.onStreamingChange).toHaveBeenCalledWith(false)
  })

  it('calls onReset when reset clicked', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    fireEvent.click(screen.getByText('Reset'))
    expect(defaultProps.onReset).toHaveBeenCalledOnce()
  })

  it('shows collapsible message length label', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    expect(screen.getByText('Auto-collapse messages longer than')).toBeTruthy()
  })
})
