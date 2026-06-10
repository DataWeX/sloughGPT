import { describe, it, expect, vi } from 'vitest'

// Mock all dependencies before importing component
vi.mock('expo-router', () => ({
  useRouter: () => ({ replace: vi.fn() }),
}))

vi.mock('@/stores/auth-store', () => ({
  useAuthStore: () => ({ login: vi.fn() }),
}))

vi.mock('@/lib/api-client', () => ({
  apiPost: vi.fn(),
}))

describe('Login Screen', () => {
  it('should export default component', async () => {
    const module = await import('@/app/(auth)/login')
    expect(module.default).toBeDefined()
    expect(typeof module.default).toBe('function')
  })

  it('should be a React component', async () => {
    const module = await import('@/app/(auth)/login')
    expect(module.default.name).toBe('LoginScreen')
  })
})
