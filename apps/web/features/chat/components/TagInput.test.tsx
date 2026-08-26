import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { TagInput } from './TagInput'

afterEach(cleanup)

describe('TagInput', () => {
  it('renders input with placeholder', () => {
    render(<TagInput tags={[]} onAdd={vi.fn()} onRemove={vi.fn()} />)
    expect(screen.getByPlaceholderText('Add tag...')).toBeInTheDocument()
  })

  it('renders existing tags', () => {
    render(<TagInput tags={['important', 'work']} onAdd={vi.fn()} onRemove={vi.fn()} />)
    expect(screen.getByText('important')).toBeInTheDocument()
    expect(screen.getByText('work')).toBeInTheDocument()
  })

  it('calls onAdd when Enter pressed', () => {
    const onAdd = vi.fn()
    render(<TagInput tags={[]} onAdd={onAdd} onRemove={vi.fn()} />)
    
    const input = screen.getByPlaceholderText('Add tag...')
    fireEvent.change(input, { target: { value: 'new tag' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    
    expect(onAdd).toHaveBeenCalledWith('new tag')
  })

  it('calls onRemove when X clicked', () => {
    const onRemove = vi.fn()
    render(<TagInput tags={['important']} onAdd={vi.fn()} onRemove={onRemove} />)
    
    fireEvent.click(screen.getByLabelText('Remove important'))
    
    expect(onRemove).toHaveBeenCalledWith('important')
  })

  it('calls onRemove on Backspace when input empty', () => {
    const onRemove = vi.fn()
    render(<TagInput tags={['important']} onAdd={vi.fn()} onRemove={onRemove} />)
    
    const input = screen.getByPlaceholderText('')
    fireEvent.keyDown(input, { key: 'Backspace' })
    
    expect(onRemove).toHaveBeenCalledWith('important')
  })

  it('shows add button when input has value', () => {
    render(<TagInput tags={[]} onAdd={vi.fn()} onRemove={vi.fn()} />)
    
    const input = screen.getByPlaceholderText('Add tag...')
    fireEvent.change(input, { target: { value: 'new tag' } })
    
    expect(screen.getByLabelText('Add tag')).toBeInTheDocument()
  })

  it('hides add button when input is empty', () => {
    render(<TagInput tags={[]} onAdd={vi.fn()} onRemove={vi.fn()} />)
    expect(screen.queryByLabelText('Add tag')).not.toBeInTheDocument()
  })

  it('disables input when disabled prop is true', () => {
    render(<TagInput tags={['important']} onAdd={vi.fn()} onRemove={vi.fn()} disabled />)
    expect(screen.queryByLabelText('Remove important')).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText('Add tag...')).not.toBeInTheDocument()
  })
})