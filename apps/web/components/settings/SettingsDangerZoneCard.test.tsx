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
  AlertDialog: ({ children }: any) => <div>{children}</div>,
  AlertDialogTrigger: ({ children }: any) => <div>{children}</div>,
  AlertDialogContent: ({ children }: any) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
  AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
  AlertDialogCancel: ({ children }: any) => <button>{children}</button>,
  AlertDialogAction: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { SettingsDangerZoneCard } from './SettingsDangerZoneCard'

afterEach(() => cleanup())

describe('SettingsDangerZoneCard', () => {
  it('renders title', () => {
    render(<SettingsDangerZoneCard onClearChat={() => {}} onResetSettings={() => {}} />)
    expect(screen.getByText('Danger zone')).toBeTruthy()
  })

  it('renders clear chat button', () => {
    render(<SettingsDangerZoneCard onClearChat={() => {}} onResetSettings={() => {}} />)
    expect(screen.getByText('Clear chat history')).toBeTruthy()
    expect(screen.getByText('Clear')).toBeTruthy()
  })

  it('renders reset settings button', () => {
    render(<SettingsDangerZoneCard onClearChat={() => {}} onResetSettings={() => {}} />)
    expect(screen.getByText('Reset all settings')).toBeTruthy()
    expect(screen.getByText('Reset')).toBeTruthy()
  })
})
