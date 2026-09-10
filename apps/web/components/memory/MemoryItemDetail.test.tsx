// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { MemoryItemDetail } from './MemoryItemDetail'

afterEach(() => { cleanup() })

const item = {
  id: 'mem-1',
  topic: 'User preferences',
  content: 'Alice prefers dark mode and uses Vim keybindings.',
  source: 'conversation',
  importance: 0.85,
  timestamp: '2025-06-15T10:30:00Z',
}

describe('MemoryItemDetail', () => {
  it('renders placeholder when no item selected', () => {
    render(<MemoryItemDetail item={null} />)
    expect(screen.getByText('Click a memory item to view details.')).toBeDefined()
  })

  it('renders Select item title when no item', () => {
    render(<MemoryItemDetail item={null} />)
    expect(screen.getByText('Select item')).toBeDefined()
  })

  it('renders item topic as title', () => {
    render(<MemoryItemDetail item={item} />)
    expect(screen.getByText('User preferences')).toBeDefined()
  })

  it('renders item content', () => {
    render(<MemoryItemDetail item={item} />)
    expect(screen.getByText(/Alice prefers dark mode/)).toBeDefined()
  })

  it('renders Edit and Delete buttons', () => {
    render(<MemoryItemDetail item={item} />)
    expect(screen.getByText('Edit')).toBeDefined()
    expect(screen.getByText('Delete')).toBeDefined()
  })

  it('shows edit textarea in edit mode', () => {
    render(<MemoryItemDetail item={item} editMode editContent={item.content} />)
    expect(screen.getByLabelText('Edit memory content')).toBeDefined()
  })

  it('shows Save and Cancel in edit mode', () => {
    render(<MemoryItemDetail item={item} editMode editContent={item.content} />)
    expect(screen.getByText('Save')).toBeDefined()
    expect(screen.getByText('Cancel')).toBeDefined()
  })
})
