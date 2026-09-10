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
  Input: (props: any) => <input {...props} />,
}))

vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://localhost:8000' }))

import { SettingsConnectionCard } from './SettingsConnectionCard'

afterEach(() => cleanup())

describe('SettingsConnectionCard', () => {
  it('renders title', () => {
    render(<SettingsConnectionCard apiUrl="" hfToken="" onApiUrlChange={() => {}} onHfTokenChange={() => {}} />)
    expect(screen.getByText('Connection')).toBeTruthy()
  })

  it('renders API URL input', () => {
    render(<SettingsConnectionCard apiUrl="http://localhost:8000" hfToken="" onApiUrlChange={() => {}} onHfTokenChange={() => {}} />)
    expect(screen.getByDisplayValue('http://localhost:8000')).toBeTruthy()
  })

  it('renders HF token input', () => {
    render(<SettingsConnectionCard apiUrl="" hfToken="hf_abc" onApiUrlChange={() => {}} onHfTokenChange={() => {}} />)
    expect(screen.getByDisplayValue('hf_abc')).toBeTruthy()
  })

  it('renders test connection button', () => {
    render(<SettingsConnectionCard apiUrl="" hfToken="" onApiUrlChange={() => {}} onHfTokenChange={() => {}} />)
    expect(screen.getByText('Test connection')).toBeTruthy()
  })
})
