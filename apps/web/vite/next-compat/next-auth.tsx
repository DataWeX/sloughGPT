/** next-auth/react Phase-0 stub — real auth lives in FastAPI /login. */
import type { ReactNode } from 'react'

export type SessionProviderProps = {
  children: ReactNode
  session?: unknown
  baseUrl?: string
  basePath?: string
  refetchInterval?: number
  refetchOnWindowFocus?: boolean
  refetchWhenOffline?: boolean
}

export function SessionProvider({ children }: SessionProviderProps) {
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
