import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'

import VisualDatasetCard from '@/components/multimodal/VisualDatasetCard'

afterEach(cleanup)

describe('VisualDatasetCard', () => {
  const onCreate = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders title', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    expect(screen.getAllByText('Image description dataset').length).toBeGreaterThanOrEqual(1)
  })

  it('renders description', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    expect(screen.getAllByText(/Create a training dataset/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders both inputs', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    expect(screen.getAllByLabelText('Visual dataset name').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByLabelText('Image directory for visual dataset').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Create button', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    expect(screen.getAllByText('Create dataset').length).toBeGreaterThanOrEqual(1)
  })

  it('disables Create when name empty', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    fireEvent.change(screen.getAllByLabelText('Image directory for visual dataset')[0], { target: { value: '/path' } })
    const btns = screen.getAllByText('Create dataset').filter(el => el.closest('button'))
    expect(btns[0].closest('button')).toBeDisabled()
  })

  it('disables Create when dir empty', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    fireEvent.change(screen.getAllByLabelText('Visual dataset name')[0], { target: { value: 'test' } })
    const btns = screen.getAllByText('Create dataset').filter(el => el.closest('button'))
    expect(btns[0].closest('button')).toBeDisabled()
  })

  it('enables Create when both filled', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    fireEvent.change(screen.getAllByLabelText('Visual dataset name')[0], { target: { value: 'my-data' } })
    fireEvent.change(screen.getAllByLabelText('Image directory for visual dataset')[0], { target: { value: '/imgs' } })
    const btns = screen.getAllByText('Create dataset').filter(el => el.closest('button'))
    expect(btns[0].closest('button')).not.toBeDisabled()
  })

  it('calls onCreate with name and dir', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    fireEvent.change(screen.getAllByLabelText('Visual dataset name')[0], { target: { value: 'my-data' } })
    fireEvent.change(screen.getAllByLabelText('Image directory for visual dataset')[0], { target: { value: '/imgs' } })
    const btns = screen.getAllByText('Create dataset').filter(el => el.closest('button'))
    fireEvent.click(btns[0].closest('button')!)
    expect(onCreate).toHaveBeenCalledWith('my-data', '/imgs')
  })

  it('trims whitespace', () => {
    render(<VisualDatasetCard creatingDataset={false} onCreate={onCreate} />)
    fireEvent.change(screen.getAllByLabelText('Visual dataset name')[0], { target: { value: '  test  ' } })
    fireEvent.change(screen.getAllByLabelText('Image directory for visual dataset')[0], { target: { value: '  /path  ' } })
    const btns = screen.getAllByText('Create dataset').filter(el => el.closest('button'))
    fireEvent.click(btns[0].closest('button')!)
    expect(onCreate).toHaveBeenCalledWith('test', '/path')
  })

  it('shows Creating when loading', () => {
    render(<VisualDatasetCard creatingDataset={true} onCreate={onCreate} />)
    expect(screen.getAllByText('Creating…').length).toBeGreaterThanOrEqual(1)
  })

  it('disables button when creating', () => {
    render(<VisualDatasetCard creatingDataset={true} onCreate={onCreate} />)
    const btns = screen.getAllByText('Creating…').filter(el => el.closest('button'))
    expect(btns[0].closest('button')).toBeDisabled()
  })
})
