import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.hoisted(() => {
  process.env.NEXTAUTH_SECRET = 'test-secret'
  ;(process.env as any).NODE_ENV = 'test'
})

const { mockGithub, mockCredentials } = vi.hoisted(() => ({
  mockGithub: vi.fn((opts: any) => ({ id: 'github', options: opts })),
  mockCredentials: vi.fn((opts: any) => ({ id: opts.id ?? 'credentials', options: opts })),
}))

vi.mock('next-auth/providers/github', () => ({ default: (opts: any) => mockGithub(opts) }))
vi.mock('next-auth/providers/credentials', () => ({ default: (opts: any) => mockCredentials(opts) }))

import { createAuthOptions } from './authOptions'

function makeEnv(overrides: Partial<Record<string, string>> = {}): NodeJS.ProcessEnv {
  return { NODE_ENV: 'test', NEXTAUTH_SECRET: 'test-secret', ...overrides } as NodeJS.ProcessEnv
}

describe('authOptions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGithub.mockImplementation((opts: any) => ({ id: 'github', options: opts }))
    mockCredentials.mockImplementation((opts: any) => ({ id: opts.id ?? 'credentials', options: opts }))
  })

  it('uses GithubProvider when github env vars are set', () => {
    const opts = createAuthOptions(makeEnv({ GITHUB_ID: 'gh-id', GITHUB_SECRET: 'gh-secret' }))
    expect(mockGithub).toHaveBeenCalledWith({ clientId: 'gh-id', clientSecret: 'gh-secret' })
    expect(opts.providers.length).toBe(1)
  })

  it('falls back to CredentialsProvider without github env', () => {
    const opts = createAuthOptions(makeEnv())
    expect(mockCredentials).toHaveBeenCalled()
    expect(opts.providers.length).toBe(1)
  })

  it('uses the provided NEXTAUTH_SECRET', () => {
    const opts = createAuthOptions(makeEnv({ NEXTAUTH_SECRET: 'explicit-secret' }))
    expect(opts.secret).toBe('explicit-secret')
  })

  it('throws when NEXTAUTH_SECRET is empty and NODE_ENV is not development', () => {
    expect(() => createAuthOptions(makeEnv({ NEXTAUTH_SECRET: '' }))).toThrow()
  })

  it('uses a dev-only fallback secret in development', () => {
    const opts = createAuthOptions(makeEnv({ NEXTAUTH_SECRET: '', NODE_ENV: 'development' }))
    expect(opts.secret).toBe('development-only-change-me')
  })

  it('sets the sign-in page to /login/', () => {
    const opts = createAuthOptions(makeEnv())
    expect(opts.pages).toEqual({ signIn: '/login/' })
  })

  it('copies token.sub into session.user.name in the session callback', async () => {
    const opts = createAuthOptions(makeEnv())
    const session = { user: { name: 'old' }, expires: 'never' }
    const result = await opts.callbacks!.session!({
      session: session as any,
      token: { sub: 'user-42' } as any,
      user: {} as any,
      newSession: {} as any,
      trigger: 'update' as any,
    })
    expect(result.user!.name).toBe('user-42')
  })

  it('leaves the session untouched when it has no user', async () => {
    const opts = createAuthOptions(makeEnv())
    const session = { expires: 'never' }
    const result = await opts.callbacks!.session!({
      session: session as any,
      token: { sub: 'user-42' } as any,
      user: {} as any,
      newSession: {} as any,
      trigger: 'update' as any,
    })
    expect(result).toBe(session)
  })
})
