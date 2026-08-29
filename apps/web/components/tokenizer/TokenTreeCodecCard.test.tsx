import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockEncode: vi.fn(),
  mockDecode: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: { encode: mocks.mockEncode, decode: mocks.mockDecode },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mocks.mockAddToast }) => unknown) =>
    selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Textarea: (props: any) => <textarea {...props} />,
  Input: (props: any) => <input {...props} />,
  Button: ({ children, onClick, disabled }: any) => (
    <button onClick={onClick} disabled={disabled}>{children}</button>
  ),
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  IconChevronRight: () => <span />,
  IconChevronLeft: () => <span />,
}))

import { TokenTreeCodecCard } from './TokenTreeCodecCard'

describe('TokenTreeCodecCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the card with default sample text', () => {
    render(<TokenTreeCodecCard />)
    expect(screen.getByText('Token Tree Codec')).toBeDefined()
    expect(screen.getByLabelText('Text to encode')).toHaveValue('the quick brown fox jumps over the lazy dog')
    expect(screen.getByRole('button', { name: /Encode/i })).toBeDefined()
    expect(screen.getByRole('button', { name: /Decode/i })).toBeDefined()
  })

  it('encodes text and shows tokens and ids', async () => {
    mocks.mockEncode.mockResolvedValueOnce({
      tokens: ['the</w>', 'quick</w>', 'brown</w>'],
      ids: [3, 12, 45],
    })
    render(<TokenTreeCodecCard />)

    fireEvent.change(screen.getByLabelText('Text to encode'), { target: { value: 'the quick brown' } })
    fireEvent.click(screen.getByRole('button', { name: /Encode/i }))

    await waitFor(() => expect(mocks.mockEncode).toHaveBeenCalledWith('the quick brown'))
    expect(screen.getByText('3 tokens')).toBeDefined()
    expect(screen.getByText('the')).toBeDefined()
    expect(screen.getByText('quick')).toBeDefined()
    expect(screen.getByText('brown')).toBeDefined()
    expect(screen.getByLabelText('Token ids to decode')).toHaveValue('3, 12, 45')
  })

  it('shows an error toast when encode fails', async () => {
    mocks.mockEncode.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreeCodecCard />)
    fireEvent.click(screen.getByRole('button', { name: /Encode/i }))

    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Could not encode text', 'error'))
  })

  it('decodes ids entered by the user', async () => {
    mocks.mockDecode.mockResolvedValueOnce({ text: 'hello world' })
    render(<TokenTreeCodecCard />)

    fireEvent.change(screen.getByLabelText('Token ids to decode'), { target: { value: '3, 12 99' } })
    fireEvent.click(screen.getByRole('button', { name: /Decode/i }))

    await waitFor(() => expect(mocks.mockDecode).toHaveBeenCalledWith([3, 12, 99]))
    expect(screen.getByText('"hello world"')).toBeDefined()
  })

  it('shows an error toast when decode input is empty', async () => {
    render(<TokenTreeCodecCard />)
    fireEvent.change(screen.getByLabelText('Token ids to decode'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: /Decode/i }))

    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Enter at least one token id', 'error'))
    expect(mocks.mockDecode).not.toHaveBeenCalled()
  })

  it('shows an error toast when decode fails', async () => {
    mocks.mockDecode.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreeCodecCard />)
    fireEvent.change(screen.getByLabelText('Token ids to decode'), { target: { value: '3' } })
    fireEvent.click(screen.getByRole('button', { name: /Decode/i }))

    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Could not decode ids', 'error'))
  })
})
