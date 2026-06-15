'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { IconBrain } from '@/components/ui'
import { authController } from '@/lib/controllers'
import { useAuthStore } from '@/lib/auth'

export default function LoginPage() {
  const router = useRouter()
  const login = useAuthStore((state) => state.login)
  const [isLogin, setIsLogin] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      if (isLogin) {
        const res = await authController.login(form.username, form.password)
        if (res.token) {
          login(res.user, res.token)
          router.push('/')
        } else {
          setError('Login failed')
        }
      } else {
        const res = await authController.register(form.username, form.email, form.password)
        if (res.token) {
          login(res.user, res.token)
          router.push('/')
        } else {
          setError('Registration failed')
        }
      }
    } catch {
      setError('An error occurred. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-0 w-full flex-1 items-center justify-center overflow-y-auto overscroll-contain p-6">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-3 text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-lg border border-primary/30 bg-primary/10">
            <IconBrain className="h-7 w-7 text-primary" />
          </div>
          <CardTitle className="text-base">SloughGPT</CardTitle>
          <CardDescription>
            {isLogin ? 'Sign in to your account' : 'Create your account'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                type="text"
                value={form.username}
                onChange={(e) => setForm({ ...form, username: e.target.value })}
                required
                autoComplete="username"
              />
            </div>

            {!isLogin && (
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  required={!isLogin}
                  autoComplete="email"
                />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                required
                autoComplete={isLogin ? 'current-password' : 'new-password'}
              />
            </div>

            {error && <p className="text-sm text-destructive" role="alert" aria-live="assertive">{error}</p>}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Loading…' : isLogin ? 'Sign in' : 'Create account'}
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex flex-col gap-4 border-t border-border">
          <p className="text-center text-sm text-muted-foreground">
            {isLogin ? "Don't have an account? " : 'Already have an account? '}
            <Button
              type="button"
              variant="ghost"
              className="h-auto px-0 font-medium text-primary hover:text-primary"
              onClick={() => setIsLogin(!isLogin)}
            >
              {isLogin ? 'Sign up' : 'Sign in'}
            </Button>
          </p>
          <Separator />
          <p className="text-center text-xs text-muted-foreground">
            Local app — no authentication required in development mode
          </p>
        </CardFooter>
      </Card>
    </div>
  )
}
