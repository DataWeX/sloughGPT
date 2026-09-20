import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'

const mockReplace = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: mockReplace }),
}))

import SoulsRedirect from './page'

describe('SoulsRedirect', () => {
  it('redirects to Personalities', () => {
    render(<SoulsRedirect />)
    expect(mockReplace).toHaveBeenCalledWith('/personality')
  })
})
