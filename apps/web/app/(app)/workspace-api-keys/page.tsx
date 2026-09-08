'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPost, apiDelete } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

interface ApiKey {
  id: string
  name: string
  key_hash: string
  scopes: string[]
  created_at: number
  revoked: boolean
  expires_at?: number
  workspace_id: string
  user_id: string
}

interface KeysResponse {
  data: { keys: ApiKey[]; count: number }
}

export default function WorkspaceApiKeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [loading, setLoading] = useState(true)
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null)
  const addToast = useToastStore(s => s.addToast)
  const { currentWorkspace } = useAuthStore()

  const fetchKeys = useCallback(async () => {
    try {
      const res = await apiGet<KeysResponse>('/security/keys')
      setKeys(res?.data?.keys ?? [])
    } catch {
      logger.warning('Could not fetch API keys')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchKeys() }, [fetchKeys])

  const createKey = async () => {
    if (!newName.trim()) return
    setCreating(true)
    try {
      const res = await apiPost<{ data: ApiKey }>('/security/keys', { name: newName })
      setNewKeyValue(res?.data?.key ?? null)
      setNewName('')
      await fetchKeys()
      addToast('API key created', 'success')
    } catch {
      addToast('Could not create API key', 'error')
    } finally {
      setCreating(false)
    }
  }

  const revokeKey = async (keyId: string) => {
    try {
      await apiDelete(`/security/keys/${keyId}`)
      await fetchKeys()
      addToast('API key revoked', 'success')
    } catch {
      addToast('Could not revoke API key', 'error')
    }
  }

  const rotateKey = async (keyId: string) => {
    try {
      const res = await apiPost<{ data: ApiKey }>(`/security/keys/${keyId}/rotate`, {})
      setNewKeyValue(res?.data?.key ?? null)
      await fetchKeys()
      addToast('API key rotated', 'success')
    } catch {
      addToast('Could not rotate API key', 'error')
    }
  }

  const activeKeys = keys.filter(k => !k.revoked)
  const revokedKeys = keys.filter(k => k.revoked)

  const formatTimestamp = (ts: number) => {
    if (!ts) return '—'
    try {
      return new Date(ts * 1000).toLocaleString()
    } catch {
      return String(ts)
    }
  }

  return (
    <PageContainer>
      <AppRouteHeader>
        <AppRouteHeaderLead>Workspace API Keys</AppRouteHeaderLead>
      </AppRouteHeader>

      {loading ? (
        <div className="space-y-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-60 w-full" />
        </div>
      ) : (
        <div className="space-y-6">
          <KpiGrid>
            <StatCard label="Total Keys" value={String(keys.length)} />
            <StatCard label="Active" value={String(activeKeys.length)} />
            <StatCard label="Revoked" value={String(revokedKeys.length)} />
          </KpiGrid>

          {newKeyValue && (
            <Card className="border-green-500/30 bg-green-500/5">
              <CardHeader>
                <CardTitle className="text-green-600 dark:text-green-400">Key Created</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground mb-2">
                  Copy this key now. It will not be shown again.
                </p>
                <code className="block p-3 rounded bg-muted text-xs font-mono break-all">
                  {newKeyValue}
                </code>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-3"
                  onClick={() => {
                    navigator.clipboard.writeText(newKeyValue)
                    addToast('Copied to clipboard', 'success')
                  }}
                >
                  Copy
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-3 ml-2"
                  onClick={() => setNewKeyValue(null)}
                >
                  Dismiss
                </Button>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Create API Key</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  value={newName}
                  onChange={e => setNewName(e.target.value)}
                  placeholder="Key name"
                  className="flex-1"
                />
                <Button
                  onClick={createKey}
                  disabled={!newName.trim() || creating}
                  size="sm"
                >
                  {creating ? 'Creating...' : 'Create'}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Active Keys</CardTitle>
            </CardHeader>
            <CardContent>
              {activeKeys.length === 0 ? (
                <p className="text-sm text-muted-foreground">No active API keys.</p>
              ) : (
                <div className="space-y-3">
                  {activeKeys.map(key => (
                    <div
                      key={key.id}
                      className="flex items-center justify-between rounded-lg border border-border/50 p-3"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">{key.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {key.key_hash} · Created {formatTimestamp(key.created_at)}
                          {key.expires_at ? ` · Expires ${formatTimestamp(key.expires_at)}` : ''}
                        </p>
                        <div className="flex gap-1 mt-1">
                          {key.scopes.map(s => (
                            <span
                              key={s}
                              className="inline-block rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary"
                            >
                              {s}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="flex gap-1 ml-3">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => rotateKey(key.id)}
                        >
                          Rotate
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => revokeKey(key.id)}
                        >
                          Revoke
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {revokedKeys.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Revoked Keys</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {revokedKeys.map(key => (
                    <div
                      key={key.id}
                      className="flex items-center justify-between rounded-lg border border-border/30 p-3 opacity-60"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">{key.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {key.key_hash} · Created {formatTimestamp(key.created_at)}
                        </p>
                      </div>
                      <span className="text-xs text-destructive ml-3">Revoked</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </PageContainer>
  )
}
