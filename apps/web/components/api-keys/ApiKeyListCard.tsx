'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button, Badge, cn } from '@sloughgpt/strui'

interface ApiKey {
  id: string
  name: string
  key_hash: string
  scopes: string[]
  created_at: number
  expires_at?: number
  revoked: boolean
}

interface ApiKeyListCardProps {
  keys: ApiKey[]
  onRotate?: (id: string) => void
  onRevoke?: (id: string) => void
}

export function ApiKeyListCard({ keys, onRotate, onRevoke }: ApiKeyListCardProps) {
  const [confirmRevoke, setConfirmRevoke] = useState<string | null>(null)

  const activeKeys = keys.filter(k => !k.revoked)
  const revokedKeys = keys.filter(k => k.revoked)

  return (
    <Card data-testid="api-key-list">
      <CardHeader>
        <CardTitle className="text-base">
          API Keys
          <span className="text-muted-foreground font-normal ml-2">({activeKeys.length} active)</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {activeKeys.length === 0 && revokedKeys.length === 0 ? (
          <div className="text-sm text-muted-foreground">No API keys.</div>
        ) : (
          <div className="space-y-4">
            {activeKeys.length > 0 && (
              <div className="space-y-1.5">
                {activeKeys.map(key => (
                  <div key={key.id} className="flex items-center justify-between p-2.5 rounded border border-border hover:border-primary/30 transition-colors">
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-medium">{key.name}</div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <code className="text-[10px] text-muted-foreground font-mono">{key.key_hash}</code>
                        <span className="text-[9px] text-muted-foreground">
                          Created {new Date(key.created_at * 1000).toLocaleDateString()}
                        </span>
                        {key.expires_at && (
                          <span className="text-[9px] text-warning">
                            Expires {new Date(key.expires_at * 1000).toLocaleDateString()}
                          </span>
                        )}
                        {key.scopes.length > 0 && key.scopes[0] !== '*' && (
                          <span className="text-[9px] text-muted-foreground">scopes: {key.scopes.join(', ')}</span>
                        )}
                      </div>
                    </div>
                    <div className="flex gap-0.5 shrink-0">
                      {onRotate && (
                        <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => onRotate(key.id)}>
                          Rotate
                        </Button>
                      )}
                      {onRevoke && (
                        confirmRevoke === key.id ? (
                          <div className="flex gap-0.5">
                            <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={() => { onRevoke(key.id); setConfirmRevoke(null) }}>
                              Confirm
                            </Button>
                            <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setConfirmRevoke(null)}>
                              Cancel
                            </Button>
                          </div>
                        ) : (
                          <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={() => setConfirmRevoke(key.id)}>
                            Revoke
                          </Button>
                        )
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {revokedKeys.length > 0 && (
              <div>
                <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">Revoked</div>
                <div className="space-y-1">
                  {revokedKeys.map(key => (
                    <div key={key.id} className="flex items-center justify-between p-2 rounded border border-border opacity-60">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium">{key.name}</span>
                        <code className="text-[10px] text-muted-foreground font-mono">{key.key_hash}</code>
                      </div>
                      <Badge variant="secondary">Revoked</Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
