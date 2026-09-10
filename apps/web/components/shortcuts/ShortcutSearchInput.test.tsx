import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ShortcutSearchInput } from './ShortcutSearchInput'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Input: (props: any) => <input data-testid="search-input" {...props} />,
}))

vi.mock('lucide-react', () => ({
  Search: () => <span data-testid="search-icon" />,
}))

describe('ShortcutSearchInput', () => {
  it('renders the search input', () => {
    render(<ShortcutSearchInput value="" onChange={() => {}} />)
    expect(screen.getByTestId('search-input')).toBeDefined()
  })

  it('displays the current value', () => {
    render(<ShortcutSearchInput value="test query" onChange={() => {}} />)
    expect((screen.getByTestId('search-input') as HTMLInputElement).value).toBe('test query')
  })

  it('calls onChange when typing', async () => {
    const onChange = vi.fn()
    render(<ShortcutSearchInput value="" onChange={onChange} />)
    await userEvent.type(screen.getByTestId('search-input'), 'a')
    expect(onChange).toHaveBeenCalledWith('a')
  })

  it('renders the search icon', () => {
    render(<ShortcutSearchInput value="" onChange={() => {}} />)
    expect(screen.getByTestId('search-icon')).toBeDefined()
  })

  it('has the correct placeholder', () => {
    render(<ShortcutSearchInput value="" onChange={() => {}} />)
    expect(screen.getByTestId('search-input').getAttribute('placeholder')).toBe('Search shortcuts...')
  })

  it('accepts custom placeholder', () => {
    render(<ShortcutSearchInput value="" onChange={() => {}} placeholder="Custom" />)
    expect(screen.getByTestId('search-input').getAttribute('placeholder')).toBe('Custom')
  })
})
