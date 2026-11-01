import { describe, it, expect, vi } from 'vitest'
import { redirect } from 'next/navigation'

vi.mock('next/navigation', () => ({
  redirect: vi.fn(),
}))

import ErrorsPage from './page'

describe('ErrorsPage', () => {
  it('redirects to /monitoring', () => {
    ErrorsPage()
    expect(redirect).toHaveBeenCalledWith('/monitoring')
  })
})
