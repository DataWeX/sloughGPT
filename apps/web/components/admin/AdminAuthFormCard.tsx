'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface AdminAuthFormCardProps {
  mode: 'login' | 'register'
  username: string
  email: string
  password: string
  loading: boolean
  error: string | null
  onModeChange: () => void
  onUsernameChange: (value: string) => void
  onEmailChange: (value: string) => void
  onPasswordChange: (value: string) => void
  onSubmit: () => void
}

export function AdminAuthFormCard({
  mode,
  username,
  email,
  password,
  loading,
  error,
  onModeChange,
  onUsernameChange,
  onEmailChange,
  onPasswordChange,
  onSubmit,
}: AdminAuthFormCardProps) {
  const title = mode === 'login' ? 'Login' : 'Register'
  const submitLabel = loading ? (mode === 'login' ? 'Logging in...' : 'Registering...') : title

  return (
    <Card data-testid="admin-auth-form">
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {error && (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-[11px] text-destructive" data-testid="auth-error">
            {error}
          </div>
        )}

        <div className="space-y-2">
          <Input
            placeholder="Username"
            value={username}
            onChange={(e) => onUsernameChange(e.target.value)}
            aria-label="Username"
          />
          {mode === 'register' && (
            <Input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => onEmailChange(e.target.value)}
              aria-label="Email"
            />
          )}
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => onPasswordChange(e.target.value)}
            aria-label="Password"
          />
        </div>

        <Button
          className="w-full"
          onClick={onSubmit}
          disabled={loading}
        >
          {submitLabel}
        </Button>

        <p className="text-center text-[11px] text-muted-foreground">
          {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}{' '}
          <button
            type="button"
            className="text-primary underline hover:text-primary/80"
            onClick={onModeChange}
          >
            {mode === 'login' ? 'Register' : 'Login'}
          </button>
        </p>
      </CardContent>
    </Card>
  )
}
