// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { CollectionCreateCard } from './CollectionCreateCard'

afterEach(() => cleanup())

describe('CollectionCreateCard', () => {
  it('renders create button', () => {
    render(<CollectionCreateCard />)
    expect(screen.getByText('New Pipeline')).toBeTruthy()
    expect(screen.getByText('+ Create')).toBeTruthy()
  })

  it('opens form on click', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    expect(screen.getByPlaceholderText('my-pipeline')).toBeTruthy()
    expect(screen.getByText('Create Pipeline')).toBeTruthy()
    expect(screen.getByText('Cancel')).toBeTruthy()
  })

  it('closes form on cancel', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByPlaceholderText('my-pipeline')).toBeNull()
  })

  it('disables create when empty', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    const createBtn = screen.getByText('Create Pipeline')
    expect(createBtn.hasAttribute('disabled')).toBe(true)
  })

  it('enables create when name entered', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    fireEvent.change(screen.getByPlaceholderText('my-pipeline'), { target: { value: 'test-pipe' } })
    const createBtn = screen.getByText('Create Pipeline')
    expect(createBtn.hasAttribute('disabled')).toBe(false)
  })

  it('calls onCreate with values', async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined)
    render(<CollectionCreateCard onCreate={onCreate} />)
    fireEvent.click(screen.getByText('+ Create'))
    fireEvent.change(screen.getByPlaceholderText('my-pipeline'), { target: { value: 'my-pipe' } })
    fireEvent.click(screen.getByText('Create Pipeline'))
    await waitFor(() => {
      expect(onCreate).toHaveBeenCalledWith('my-pipe', 'file', 'memory')
    })
  })

  it('shows source and store selects', () => {
    render(<CollectionCreateCard />)
    fireEvent.click(screen.getByText('+ Create'))
    const selects = screen.getAllByRole('combobox')
    expect(selects.length).toBe(2)
  })
})
