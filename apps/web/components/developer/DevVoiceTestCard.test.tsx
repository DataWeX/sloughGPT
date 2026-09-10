// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Textarea: (props: any) => <textarea {...props} />,
}))

import { DevVoiceTestCard } from './DevVoiceTestCard'

afterEach(() => cleanup())

describe('DevVoiceTestCard', () => {
  it('renders voice status card', () => {
    render(<DevVoiceTestCard />)
    expect(screen.getByText('Voice Status')).toBeTruthy()
  })

  it('renders quick test card', () => {
    render(<DevVoiceTestCard />)
    expect(screen.getByText('Quick Test')).toBeTruthy()
  })

  it('renders TTS input', () => {
    render(<DevVoiceTestCard />)
    expect(screen.getByLabelText('Text to speech input')).toBeTruthy()
  })

  it('renders speak button', () => {
    render(<DevVoiceTestCard />)
    expect(screen.getByRole('button', { name: /Speak/ })).toBeTruthy()
  })

  it('speak button disabled when empty', () => {
    render(<DevVoiceTestCard />)
    expect(screen.getByRole('button', { name: /Speak/ }).getAttribute('disabled')).not.toBeNull()
  })
})
