'use client'

import { useMemo } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface ApiKey {
  id: string
  name: string
  key_hash: string
  scopes: string[]
  created_at: number
  revoked: boolean
}

interface PermissionMatrixCardProps {
  keys: ApiKey[]
}

const ALL_SCOPES = ['read', 'write', 'admin', 'train', 'delete', 'export', 'infer', 'upload', 'audit', 'manage']
const WILDCARD = '*'

function formatTimestamp(ts: number): string {
  try {
    return new Date(ts * 1000).toLocaleDateString()
  } catch {
    return '—'
  }
}

export function PermissionMatrixCard({ keys }: PermissionMatrixCardProps) {
  const activeKeys = useMemo(() => keys.filter(k => !k.revoked), [keys])
  const allScopes = useMemo(() => {
    const scopeSet = new Set<string>()
    for (const k of activeKeys) {
      for (const s of k.scopes) {
        if (s === WILDCARD) {
          ALL_SCOPES.forEach(sc => scopeSet.add(sc))
        } else {
          scopeSet.add(s)
        }
      }
    }
    return Array.from(scopeSet).sort()
  }, [activeKeys])

  const scopeKeyCount = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const scope of allScopes) {
      counts[scope] = activeKeys.filter(k =>
        k.scopes.includes(WILDCARD) || k.scopes.includes(scope)
      ).length
    }
    return counts
  }, [allScopes, activeKeys])

  if (activeKeys.length === 0) {
    return (
      <Card data-testid="permission-matrix">
        <CardHeader><CardTitle className="text-base">Permission Matrix</CardTitle></CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground text-center py-3">No active API keys.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="permission-matrix">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Permission Matrix</CardTitle>
          <span className="text-[10px] text-muted-foreground">{activeKeys.length} key{activeKeys.length !== 1 ? 's' : ''} · {allScopes.length} scope{allScopes.length !== 1 ? 's' : ''}</span>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="border-b border-border/40">
                <th className="text-left py-1.5 pr-2 text-muted-foreground font-medium">Key</th>
                {allScopes.map(scope => (
                  <th key={scope} className="text-center py-1.5 px-1 text-muted-foreground font-medium min-w-[40px]">
                    {scope}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {activeKeys.map(key => (
                <tr key={key.id} className="border-b border-border/20 last:border-0">
                  <td className="py-1.5 pr-2">
                    <div className="font-medium truncate max-w-[120px]">{key.name}</div>
                    <div className="text-[9px] text-muted-foreground/60">{formatTimestamp(key.created_at)}</div>
                  </td>
                  {allScopes.map(scope => {
                    const has = key.scopes.includes(WILDCARD) || key.scopes.includes(scope)
                    return (
                      <td key={scope} className="text-center py-1.5 px-1">
                        <span className={cn(
                          'inline-block h-3.5 w-3.5 rounded-sm',
                          has ? 'bg-success' : 'bg-muted/50'
                        )} title={has ? `${key.name} has ${scope}` : `${key.name} lacks ${scope}`} />
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t border-border/40">
                <td className="py-1.5 pr-2 text-muted-foreground font-medium">Keys</td>
                {allScopes.map(scope => (
                  <td key={scope} className="text-center py-1.5 px-1 text-[9px] text-muted-foreground">
                    {scopeKeyCount[scope]}/{activeKeys.length}
                  </td>
                ))}
              </tr>
            </tfoot>
          </table>
        </div>

        <div className="flex items-center gap-3 text-[9px] text-muted-foreground">
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-success" /> Granted</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-muted/50" /> Denied</span>
          {activeKeys.some(k => k.scopes.includes(WILDCARD)) && (
            <span className="text-warning">* Wildcard scope active</span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
