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
}))

vi.mock('@/lib/time-ago', () => ({
  timeAgo: (ts: number) => '2h ago',
}))

import { ImageHistoryCard, recordImageGeneration } from './ImageHistoryCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('ImageHistoryCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<ImageHistoryCard />)
    expect(container.firstChild).toBeNull()
  })

  it('shows history entries', async () => {
    recordImageGeneration('A cat in space', 'realistic')
    recordImageGeneration('Sunset over mountains', 'anime')
    render(<ImageHistoryCard />)
    await waitFor(() => {
      expect(screen.getByText('Generation History')).toBeTruthy()
    })
    expect(screen.getByText('A cat in space')).toBeTruthy()
    expect(screen.getByText('Sunset over mountains')).toBeTruthy()
    expect(screen.getByText('(2)')).toBeTruthy()
  })

  it('filters by style', async () => {
    recordImageGeneration('Cat', 'realistic')
    recordImageGeneration('Dog', 'anime')
    render(<ImageHistoryCard />)
    await waitFor(() => { expect(screen.getByText('Cat')).toBeTruthy() })
    const animeBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('anime'))
    if (animeBtn) fireEvent.click(animeBtn)
    expect(screen.queryByText('Cat')).toBeNull()
    expect(screen.getByText('Dog')).toBeTruthy()
  })

  it('calls onReUse', async () => {
    const onReUse = vi.fn()
    recordImageGeneration('Test prompt', 'realistic')
    render(<ImageHistoryCard onReUse={onReUse} />)
    await waitFor(() => { expect(screen.getByText('Test prompt')).toBeTruthy() })
    const reUseBtn = screen.getAllByRole('button').find(b => b.textContent === 'Re-use')
    if (reUseBtn) fireEvent.click(reUseBtn)
    expect(onReUse).toHaveBeenCalledWith('Test prompt', 'realistic')
  })

  it('deletes an entry', async () => {
    recordImageGeneration('To delete', 'realistic')
    render(<ImageHistoryCard />)
    await waitFor(() => { expect(screen.getByText('To delete')).toBeTruthy() })
    const delBtn = screen.getAllByRole('button').find(b => b.textContent === 'Del')
    if (delBtn) fireEvent.click(delBtn)
    await waitFor(() => {
      expect(screen.queryByText('To delete')).toBeNull()
    })
  })

  it('clears all history', async () => {
    recordImageGeneration('One', 'realistic')
    recordImageGeneration('Two', 'realistic')
    render(<ImageHistoryCard />)
    await waitFor(() => { expect(screen.getByText('One')).toBeTruthy() })
    fireEvent.click(screen.getByText('Clear'))
    await waitFor(() => {
      expect(screen.queryByText('One')).toBeNull()
    })
  })
})
