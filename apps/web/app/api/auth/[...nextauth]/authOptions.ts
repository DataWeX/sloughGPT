import { NextAuthOptions } from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'
import GithubProvider from 'next-auth/providers/github'

const githubId = process.env.GITHUB_ID?.trim()
const githubSecret = process.env.GITHUB_SECRET?.trim()

// NextAuth requires at least one provider. GitHub is optional; the app login UI uses FastAPI (`/login`).
const githubOrPlaceholder: NextAuthOptions['providers'] =
  githubId && githubSecret
    ? [
        GithubProvider({
          clientId: githubId,
          clientSecret: githubSecret,
        }),
      ]
    : [
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

export const authOptions: NextAuthOptions = {
  // Required for JWT/session; without it, NextAuth often fails at runtime in dev/prod.
  secret:
    process.env.NEXTAUTH_SECRET ||
    (process.env.NODE_ENV === 'development' ? 'development-only-change-me' : undefined),
  providers: githubOrPlaceholder,
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
