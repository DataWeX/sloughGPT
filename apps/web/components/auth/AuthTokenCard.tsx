'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const base64 = token.split('.')[1]
    if (!base64) return null
    const json = atob(base64.replace(/-/g, '+').replace(/_/g, '/'))
    return JSON.parse(json)
  } catch { return null }
}

function formatExpiry(exp: number): string {
  const diff = exp * 1000 - Date.now()
  if (diff <= 0) return 'Expired'
  if (diff < 60000) return `${Math.floor(diff / 1000)}s left`
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m left`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h left`
  return `${Math.floor(diff / 86400000)}d left`
}

interface AuthTokenCardProps {
  token: string | null
  onVerify?: (token: string) => Promise<{ valid: boolean; expires?: number }>
}

export function AuthTokenCard({ token, onVerify }: AuthTokenCardProps) {
  const [verifyResult, setVerifyResult] = useState<{ valid: boolean; expires?: number } | null>(null)
  const [verifying, setVerifying] = useState(false)
  const [copied, setCopied] = useState(false)

  if (!token) return null

  const payload = decodeJwtPayload(token)
  const exp = payload?.exp as number | undefined
  const iat = payload?.iat as number | undefined
  const sub = payload?.sub as string | undefined
  const role = payload?.role as string | undefined

  const handleCopy = async () => {
    await navigator.clipboard.writeText(token)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleVerify = async () => {
    if (!onVerify) return
    setVerifying(true)
    try {
      const result = await onVerify(token)
      setVerifyResult(result)
    } catch {
      setVerifyResult({ valid: false })
    } finally {
      setVerifying(false)
    }
  }

  return (
    <Card data-testid="auth-token">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Token Details</CardTitle>
          <div className="flex gap-1">
            {onVerify && (
              <Button size="sm" variant="ghost" className="text-[10px]" onClick={handleVerify} disabled={verifying}>
                {verifying ? 'Verifying...' : 'Verify'}
              </Button>
            )}
            <Button size="sm" variant="ghost" className="text-[10px]" onClick={handleCopy}>
              {copied ? 'Copied!' : 'Copy'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {sub && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Subject</div>
              <div className="text-xs font-medium mt-0.5">{sub}</div>
            </div>
          )}
          {role && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Role</div>
              <div className="text-xs font-medium mt-0.5 capitalize">{role}</div>
            </div>
          )}
          {exp && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground">Expires:</span>
              <span className={cn('text-[10px] font-mono',
                exp * 1000 < Date.now() ? 'text-destructive' :
                exp * 1000 - Date.now() < 3600000 ? 'text-warning' :
                'text-success'
              )}>
                {formatExpiry(exp)}
              </span>
            </div>
          )}
          {iat && (
            <div className="text-[10px] text-muted-foreground">
              Issued: {new Date(iat * 1000).toLocaleString()}
            </div>
          )}
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-0.5">Raw</div>
            <div className="text-[9px] font-mono text-muted-foreground break-all bg-muted/50 rounded p-1.5 max-h-16 overflow-y-auto">
              {token}
            </div>
          </div>
          {verifyResult && (
            <div className={cn(
              'text-xs font-medium p-2 rounded',
              verifyResult.valid ? 'bg-success/10 text-success' : 'bg-destructive/10 text-destructive'
            )}>
              {verifyResult.valid ? '✓ Token is valid' : '✗ Token is invalid or expired'}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
