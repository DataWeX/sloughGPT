import { describe, it, expect, vi, afterEach } from 'vitest'
import crypto from 'crypto'

const { mockGithub, mockCredentials, mockNextAuth } = vi.hoisted(() => ({
  mockGithub: vi.fn((opts: any) => ({ id: 'github', options: opts })),
  mockCredentials: vi.fn((opts: any) => ({ id: opts.id ?? 'credentials', options: opts })),
  mockNextAuth: vi.fn((opts: any) => ({ options: opts })),
}))

vi.mock('next-auth', () => ({ default: (opts: any) => mockNextAuth(opts) }))
vi.mock('next-auth/providers/github', () => ({ default: (opts: any) => mockGithub(opts) }))
vi.mock('next-auth/providers/credentials', () => ({ default: (opts: any) => mockCredentials(opts) }))

async function loadAuthOptions() {
  vi.resetModules()
  const mod = await import('./authOptions')
  return mod.authOptions
}

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('authOptions', () => {
  it('uses GithubProvider when credentials are configured', async () => {
    vi.stubEnv('GITHUB_ID', 'gh-id')
    vi.stubEnv('GITHUB_SECRET', 'gh-secret')
    const opts = await loadAuthOptions()
    expect(mockGithub).toHaveBeenCalledWith({ clientId: 'gh-id', clientSecret: 'gh-secret' })
    expect(opts.providers.length).toBe(1)
    expect(opts.providers[0].id).toBe('github')
  })

  it('falls back to placeholder CredentialsProvider without github env', async () => {
    vi.stubEnv('GITHUB_ID', '')
    vi.stubEnv('GITHUB_SECRET', '')
    const opts = await loadAuthOptions()
    expect(mockCredentials).toHaveBeenCalled()
    expect(opts.providers.length).toBe(1)
    expect(opts.providers[0].id).toBe('fastapi-login')
  })

  it('uses the provided NEXTAUTH_SECRET', async () => {
    vi.stubEnv('NEXTAUTH_SECRET', 'explicit-secret')
    const opts = await loadAuthOptions()
    expect(opts.secret).toBe('explicit-secret')
  })

  it('derives a production secret from a hash when unset and not development', async () => {
    vi.stubEnv('NEXTAUTH_SECRET', '')
    vi.stubEnv('NODE_ENV', 'test')
    const opts = await loadAuthOptions()
    const expected = crypto.createHash('sha256').update('sloughgpt-nextauth-secret').digest('base64')
    expect(opts.secret).toBe(expected)
  })

  it('uses a dev-only fallback secret in development', async () => {
    vi.stubEnv('NEXTAUTH_SECRET', '')
    vi.stubEnv('NODE_ENV', 'development')
    const opts = await loadAuthOptions()
    expect(opts.secret).toBe('development-only-change-me')
  })

  it('sets the sign-in page to /login/', async () => {
    vi.stubEnv('GITHUB_ID', '')
    vi.stubEnv('GITHUB_SECRET', '')
    const opts = await loadAuthOptions()
    expect(opts.pages).toEqual({ signIn: '/login/' })
  })

  it('copies token.sub into session.user.name in the session callback', async () => {
    vi.stubEnv('GITHUB_ID', '')
    vi.stubEnv('GITHUB_SECRET', '')
    const opts = await loadAuthOptions()
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
    vi.stubEnv('GITHUB_ID', '')
    vi.stubEnv('GITHUB_SECRET', '')
    const opts = await loadAuthOptions()
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
