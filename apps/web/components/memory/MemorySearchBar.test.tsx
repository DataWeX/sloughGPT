// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { MemorySearchBar } from './MemorySearchBar'

afterEach(() => { cleanup() })

describe('MemorySearchBar', () => {
  it('renders search input', () => {
    render(<MemorySearchBar query="" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} />)
    expect(screen.getByPlaceholderText('Search memory...')).toBeDefined()
  })

  it('renders Search button', () => {
    render(<MemorySearchBar query="" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} />)
    expect(screen.getByText('Search')).toBeDefined()
  })

  it('shows loading state when searching', () => {
    render(<MemorySearchBar query="test" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} searching />)
    expect(screen.getByText('Searching...')).toBeDefined()
  })

  it('renders Clear button when hasResults is true', () => {
    render(<MemorySearchBar query="test" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} hasResults />)
    expect(screen.getByText('Clear')).toBeDefined()
  })

  it('does not render Clear button when hasResults is false', () => {
    render(<MemorySearchBar query="test" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} hasResults={false} />)
    expect(screen.queryByText('Clear')).toBeNull()
  })

  it('calls onSearch when Search button is clicked', () => {
    const onSearch = vi.fn()
    render(<MemorySearchBar query="q" onQueryChange={() => {}} onSearch={onSearch} onClear={() => {}} />)
    screen.getByText('Search').click()
    expect(onSearch).toHaveBeenCalled()
  })
})
