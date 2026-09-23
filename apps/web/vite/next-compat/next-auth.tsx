/** next-auth/react Phase-0 stub — real auth lives in FastAPI /login. */
import type { ReactNode } from 'react'

export function SessionProvider({ children }: { children: ReactNode }) {
  return <>{children}</>
}

export function useSession() {
  return { data: null, status: 'unauthenticated' as const, update: async () => null }
}

export function signIn() {
  return Promise.resolve({ error: null })
}

export function signOut() {
  return Promise.resolve()
}

export function getServerSession() {
  return Promise.resolve(null)
}
