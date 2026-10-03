import { describe, it, expect, vi } from 'vitest'
import { redirect } from '@/vite/next-compat/navigation'

vi.mock('@/vite/next-compat/navigation', () => ({
  redirect: vi.fn(),
}))

import ErrorsPage from './page'

describe('ErrorsPage', () => {
  it('redirects to /monitoring', () => {
    ErrorsPage()
    expect(redirect).toHaveBeenCalledWith('/monitoring')
  })
})
