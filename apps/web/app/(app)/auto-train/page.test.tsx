import { describe, it, expect, vi } from 'vitest'
import { redirect } from 'next/navigation'

vi.mock('next/navigation', () => ({
  redirect: vi.fn(),
}))

import AutoTrainPage from './page'

describe('AutoTrainPage', () => {
  it('redirects to /training', () => {
    AutoTrainPage()
    expect(redirect).toHaveBeenCalledWith('/training')
  })
})
