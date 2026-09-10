// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
}))

import { AgentTemplateCard } from './AgentTemplateCard'

afterEach(() => cleanup())

const mockTemplates = [
  { name: 'Code Assistant', desc: 'Helps write and review code', instructions: 'Be concise', tools: ['read_file', 'write_file'] },
  { name: 'Research Bot', desc: 'Searches and summarizes info', instructions: 'Be thorough', tools: ['web_search'] },
]

describe('AgentTemplateCard', () => {
  it('renders title', () => {
    render(<AgentTemplateCard templates={[]} onSelect={vi.fn()} />)
    expect(screen.getByText('Agent Templates')).toBeTruthy()
  })

  it('shows empty state', () => {
    render(<AgentTemplateCard templates={[]} onSelect={vi.fn()} />)
    expect(screen.getByText('No templates available.')).toBeTruthy()
  })

  it('renders templates with name and description', () => {
    render(<AgentTemplateCard templates={mockTemplates} onSelect={vi.fn()} />)
    expect(screen.getByText('Code Assistant')).toBeTruthy()
    expect(screen.getByText('Helps write and review code')).toBeTruthy()
    expect(screen.getByText('Research Bot')).toBeTruthy()
    expect(screen.getByText('Searches and summarizes info')).toBeTruthy()
  })

  it('renders tool badges', () => {
    render(<AgentTemplateCard templates={mockTemplates} onSelect={vi.fn()} />)
    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('write_file')).toBeTruthy()
    expect(screen.getByText('web_search')).toBeTruthy()
  })

  it('calls onSelect when button clicked', () => {
    const onSelect = vi.fn()
    render(<AgentTemplateCard templates={mockTemplates} onSelect={onSelect} />)
    const buttons = screen.getAllByRole('button').filter(b => b.textContent === 'Select')
    fireEvent.click(buttons[0])
    expect(onSelect).toHaveBeenCalledWith(mockTemplates[0])
  })
})
