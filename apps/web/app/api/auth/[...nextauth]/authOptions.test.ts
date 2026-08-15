import { describe, it, expect, vi } from 'vitest'
import crypto from 'crypto'
import { createAuthOptions } from './authOptions'

const { mockGithub, mockCredentials } = vi.hoisted(() => ({
  mockGithub: vi.fn((opts: any) => ({ id: 'github', options: opts })),
  mockCredentials: vi.fn((opts: any) => ({ id: opts.id ?? 'credentials', options: opts })),
}))

vi.mock('next-auth/providers/github', () => ({ default: (opts: any) => mockGithub(opts) }))
vi.mock('next-auth/providers/credentials', () => ({ default: (opts: any) => mockCredentials(opts) }))

function makeEnv(overrides: Partial<NodeJS.ProcessEnv> = {}): NodeJS.ProcessEnv {
  return { NODE_ENV: 'test', ...overrides }
}

describe('authOptions', () => {
  it('uses GithubProvider when credentials are configured', () => {
    const opts = createAuthOptions(makeEnv({ GITHUB_ID: 'gh-id', GITHUB_SECRET: 'gh-secret' }))
    expect(mockGithub).toHaveBeenCalledWith({ clientId: 'gh-id', clientSecret: 'gh-secret' })
    expect(opts.providers.length).toBe(1)
    expect(opts.providers[0].id).toBe('github')
  })

  it('falls back to placeholder CredentialsProvider without github env', () => {
    const opts = createAuthOptions(makeEnv())
    expect(mockCredentials).toHaveBeenCalled()
    expect(opts.providers.length).toBe(1)
    expect(opts.providers[0].id).toBe('fastapi-login')
  })

  it('uses the provided NEXTAUTH_SECRET', () => {
    const opts = createAuthOptions(makeEnv({ NEXTAUTH_SECRET: 'explicit-secret' }))
    expect(opts.secret).toBe('explicit-secret')
  })

  it('derives a production secret from a hash when unset and not development', () => {
    const opts = createAuthOptions(makeEnv({ NEXTAUTH_SECRET: '' }))
    const expected = crypto.createHash('sha256').update('sloughgpt-nextauth-secret').digest('base64')
    expect(opts.secret).toBe(expected)
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
