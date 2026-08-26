'use client'

import { cn, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface AuthSessionInfoCardProps {
  token: string | null
  user: { id: string; username: string; email: string } | null
  onLogout: () => void
}

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const base64 = token.split('.')[1]
    if (!base64) return null
    const json = atob(base64.replace(/-/g, '+').replace(/_/g, '/'))
    return JSON.parse(json)
  } catch {
    return null
  }
}

function formatExpiry(exp: number): string {
  const diff = exp * 1000 - Date.now()
  if (diff <= 0) return 'expired'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m left`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h left`
  return `${Math.floor(diff / 86400000)}d left`
}

export function AuthSessionInfoCard({ token, user, onLogout }: AuthSessionInfoCardProps) {
  if (!token || !user) return null

  const payload = decodeJwtPayload(token)
  const exp = payload?.exp as number | undefined
  const iat = payload?.iat as number | undefined

  return (
    <Card data-testid="auth-session-info">
      <CardHeader>
        <CardTitle className="text-base">Session</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">User</div>
              <div className="text-sm font-medium mt-0.5">{user.username}</div>
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Email</div>
              <div className="text-sm font-medium mt-0.5">{user.email}</div>
            </div>
          </div>
          {exp && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground">Token expires:</span>
              <span className={cn('text-[10px] font-mono', exp * 1000 < Date.now() ? 'text-destructive' :
                exp * 1000 - Date.now() < 3600000 ? 'text-warning' :
                'text-success')}>
                {formatExpiry(exp)}
              </span>
            </div>
          )}
          {iat && (
            <div className="text-[10px] text-muted-foreground">
              Issued: {new Date(iat * 1000).toLocaleString()}
            </div>
          )}
          <button
            type="button"
            onClick={onLogout}
            className="text-xs text-destructive hover:underline"
          >
            Sign out
          </button>
        </div>
      </CardContent>
    </Card>
  )
}
