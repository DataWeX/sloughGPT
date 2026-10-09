import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'

const mockReplace = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: mockReplace }),
}))

import ModelsRedirect from './page'

describe('ModelsRedirect', () => {
  it('redirects to Developer Models tab host', () => {
    render(<ModelsRedirect />)
    expect(mockReplace).toHaveBeenCalledWith('/developer')
  })
})
