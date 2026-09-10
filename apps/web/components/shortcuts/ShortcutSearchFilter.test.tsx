/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { ShortcutSearchFilter } from './ShortcutSearchFilter'

vi.mock('@sloughgpt/strui', () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}))

afterEach(() => cleanup())

describe('ShortcutSearchFilter', () => {
  it('renders the search input', () => {
    render(<ShortcutSearchFilter categories={['Global', 'Chat']} onFilter={vi.fn()} />)
    expect(screen.getByLabelText('Filter shortcuts by category')).toBeDefined()
  })

  it('renders with a placeholder', () => {
    render(<ShortcutSearchFilter categories={[]} onFilter={vi.fn()} />)
    expect(screen.getByPlaceholderText('Filter categories...')).toBeDefined()
  })

  it('calls onFilter with all categories when query is empty', async () => {
    const onFilter = vi.fn()
    render(<ShortcutSearchFilter categories={['Global', 'Chat', 'Nav']} onFilter={onFilter} />)
    const input = screen.getByLabelText('Filter shortcuts by category')
    await userEvent.type(input, 'X')
    await userEvent.clear(input)
    expect(onFilter).toHaveBeenLastCalledWith(['Global', 'Chat', 'Nav'])
  })

  it('filters categories by query', async () => {
    const onFilter = vi.fn()
    render(<ShortcutSearchFilter categories={['Global', 'Chat', 'Navigation']} onFilter={onFilter} />)
    const input = screen.getByLabelText('Filter shortcuts by category')
    await userEvent.type(input, 'cha')
    expect(onFilter).toHaveBeenLastCalledWith(['Chat'])
  })

  it('returns empty array when no categories match', async () => {
    const onFilter = vi.fn()
    render(<ShortcutSearchFilter categories={['Global', 'Chat']} onFilter={onFilter} />)
    const input = screen.getByLabelText('Filter shortcuts by category')
    await userEvent.type(input, 'zzz')
    expect(onFilter).toHaveBeenLastCalledWith([])
  })
})
