import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'

const redirect = vi.fn((url: string) => {
  throw new Error(`NEXT_REDIRECT;${url}`)
})

vi.mock('@/vite/next-compat/navigation', () => ({
  redirect: (url: string) => redirect(url),
}))

import Page from './page'

describe('usage redirect', () => {
  it('redirects to /workspace/usage', () => {
    expect(() => render(<Page />)).toThrow('/workspace/usage')
    expect(redirect).toHaveBeenCalledWith('/workspace/usage')
  })
})
