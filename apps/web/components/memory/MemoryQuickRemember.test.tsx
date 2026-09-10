// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { MemoryQuickRemember } from './MemoryQuickRemember'

afterEach(() => { cleanup() })

describe('MemoryQuickRemember', () => {
  it('renders content input', () => {
    render(<MemoryQuickRemember content="" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} />)
    expect(screen.getByPlaceholderText('Quick remember: type a fact and press Enter')).toBeDefined()
  })

  it('renders topic input', () => {
    render(<MemoryQuickRemember content="" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} />)
    expect(screen.getByPlaceholderText('Topic (optional)')).toBeDefined()
  })

  it('renders Remember button', () => {
    render(<MemoryQuickRemember content="" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} />)
    expect(screen.getByText('Remember')).toBeDefined()
  })

  it('disables Remember button when content is empty', () => {
    render(<MemoryQuickRemember content="" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} />)
    expect(screen.getByText('Remember').hasAttribute('disabled')).toBe(true)
  })

  it('shows saving state', () => {
    render(<MemoryQuickRemember content="test" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={() => {}} remembering />)
    expect(screen.getByText('Saving...')).toBeDefined()
  })

  it('calls onRemember when Remember button is clicked', () => {
    const onRemember = vi.fn()
    render(<MemoryQuickRemember content="fact" topic="" onContentChange={() => {}} onTopicChange={() => {}} onRemember={onRemember} />)
    screen.getByText('Remember').click()
    expect(onRemember).toHaveBeenCalled()
  })
})
