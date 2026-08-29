import { NextAuthOptions } from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'
import GithubProvider from 'next-auth/providers/github'

function resolveProviders(env: NodeJS.ProcessEnv): NextAuthOptions['providers'] {
  const githubId = env.GITHUB_ID?.trim()
  const githubSecret = env.GITHUB_SECRET?.trim()

  // NextAuth requires at least one provider. GitHub is optional; the app login UI uses FastAPI (`/login`).
  if (githubId && githubSecret) {
    return [
      GithubProvider({
        clientId: githubId,
        clientSecret: githubSecret,
      }),
    ]
  }
  return [
    CredentialsProvider({
      id: 'fastapi-login',
      name: 'Sign in via FastAPI',
      credentials: {
        hint: { label: 'Use the login form at /login', type: 'text' },
      },
      async authorize() {
        return null
      },
    }),
  ]
}

function resolveSecret(env: NodeJS.ProcessEnv): string {
  if (env.NEXTAUTH_SECRET) return env.NEXTAUTH_SECRET
  if (env.NODE_ENV === 'development') return 'development-only-change-me'
  throw new Error(
    'NEXTAUTH_SECRET is required in production. ' +
    'Set it in your environment variables or .env.local file.',
  )
}

/** Build NextAuth options from an environment; defaults to process.env. */
export function createAuthOptions(env: NodeJS.ProcessEnv = process.env): NextAuthOptions {
  return {
    // Required for JWT/session; without it, NextAuth fails at runtime in production.
    secret: resolveSecret(env),
    providers: resolveProviders(env),
    pages: {
      // App Router uses `/login/` (see `app/(app)/login`); `/auth/signin` does not exist.
      signIn: '/login/',
    },
    callbacks: {
      async session({ session, token }) {
        if (session.user) {
          session.user.name = token.sub
        }
        return session
      },
    },
  }
}

export const authOptions: NextAuthOptions = createAuthOptions()
