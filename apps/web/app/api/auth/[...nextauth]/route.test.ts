import { describe, it, expect, vi } from 'vitest'

const mockNextAuth = vi.hoisted(() => vi.fn(() => ({ handler: true })))

vi.mock('next-auth', () => ({ default: mockNextAuth }))
vi.mock('./authOptions', () => ({ authOptions: { secret: 'mock-secret' } }))

describe('auth route', () => {
  it('builds the handler with authOptions', async () => {
    const route = await import('./route')
    expect(mockNextAuth).toHaveBeenCalledWith({ secret: 'mock-secret' })
    expect(route.GET).toEqual({ handler: true })
  })

  it('exports the same handler as GET and POST', async () => {
    const route = await import('./route')
    expect(route.GET).toBe(route.POST)
    expect(route.GET).toEqual({ handler: true })
  })
})
