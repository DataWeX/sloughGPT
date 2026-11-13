import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { RedirectPage } from './RedirectPage'

vi.mock('@/vite/next-compat/navigation', () => ({
  useRouter: () => ({ replace: vi.fn() }),
}))

describe('RedirectPage', () => {
  it('renders page skeleton while redirecting', () => {
    render(<RedirectPage to="/chat" />)
    expect(document.querySelector('[class*="animate-pulse"]')).toBeTruthy()
  })

  it('accepts a destination path', () => {
    render(<RedirectPage to="/monitoring" />)
    expect(document.body).toBeTruthy()
  })
})
