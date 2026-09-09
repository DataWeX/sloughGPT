'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Input,
  Skeleton,
  StatCard,
  KpiGrid,
} from '@sloughgpt/strui'
import { IconPlus, IconTrash, IconRefresh } from '@/components/icons/NavIcons'
import { apiGet, apiPost, apiDelete } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'

interface ApiKey {
  id: string
  name: string
  key_hash: string
  scopes: string[]
  created_at: number
  expires_at?: number
  revoked: boolean
  workspace_id: string
  user_id: string
}

interface KeysResponse {
  data: { keys: ApiKey[]; count: number }
}

interface CreateKeyResponse {
  data: ApiKey & { key: string }
}

export default function ApiKeysPage() {
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)

  const [keys, setKeys] = useState<ApiKey[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newKey, setNewKey] = useState<string | null>(null)
  const [revokingId, setRevokingId] = useState<string | null>(null)

  const fetchKeys = useCallback(async () => {
    try {
      const res = await apiGet<KeysResponse>('/security/keys')
      setKeys(res?.data?.keys ?? [])
    } catch {
      addToast('Could not load API keys', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    fetchKeys()
  }, [fetchKeys])

  const handleCreate = async () => {
    if (!newName.trim()) return
    setCreating(true)
    try {
      const res = await apiPost<CreateKeyResponse>('/security/keys', { name: newName.trim() })
      const key = res?.data?.key
      if (key) setNewKey(key)
      setNewName('')
      await fetchKeys()
      addToast('API key created', 'success')
    } catch {
      addToast('Could not create API key', 'error')
    } finally {
      setCreating(false)
    }
  }

  const handleRevoke = async (keyId: string) => {
    setRevokingId(keyId)
    try {
      await apiDelete(`/security/keys/${keyId}`)
      addToast('API key revoked', 'success')
      await fetchKeys()
    } catch {
      addToast('Could not revoke API key', 'error')
    } finally {
      setRevokingId(null)
    }
  }

  const handleRotate = async (keyId: string) => {
    try {
      const res = await apiPost<CreateKeyResponse>(`/security/keys/${keyId}/rotate`)
      const key = res?.data?.key
      if (key) setNewKey(key)
      addToast('API key rotated', 'success')
      await fetchKeys()
    } catch {
      addToast('Could not rotate API key', 'error')
    }
  }

  const activeKeys = keys.filter(k => !k.revoked)
  const revokedKeys = keys.filter(k => k.revoked)

  const copyKey = (key: string) => {
    navigator.clipboard.writeText(key)
    addToast('Key copied to clipboard', 'success')
  }

  if (loading) {
    return (
      <PageContainer title="API Keys">
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-64 w-full" />
      </PageContainer>
    )
  }

  return (
    <PageContainer title="API Keys">
      <AppRouteHeader left={<AppRouteHeaderLead title="API Keys" />} />

      {newKey && (
        <Card className="mb-6 border-green-500/50 bg-green-500/5">
          <CardHeader>
            <CardTitle className="text-green-600">New API Key</CardTitle>
            <CardDescription>
              Copy this key now. It won&apos;t be shown again.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <code className="flex-1 p-2 bg-muted rounded text-sm break-all">{newKey}</code>
              <Button size="sm" onClick={() => copyKey(newKey)}>Copy</Button>
              <Button size="sm" variant="ghost" onClick={() => setNewKey(null)}>Dismiss</Button>
            </div>
          </CardContent>
        </Card>
      )}

      <KpiGrid className="mb-6">
        <StatCard label="Active Keys" value={String(activeKeys.length)} />
        <StatCard label="Revoked" value={String(revokedKeys.length)} />
      </KpiGrid>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Create API Key</CardTitle>
          <CardDescription>
            Create a new API key for programmatic access to the workspace.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Input
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="Key name"
              maxLength={100}
              onKeyDown={e => e.key === 'Enter' && handleCreate()}
            />
            <Button onClick={handleCreate} disabled={!newName.trim() || creating}>
              <IconPlus className="mr-2 h-4 w-4" />
              {creating ? 'Creating...' : 'Create'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Active Keys</CardTitle>
          <CardDescription>{activeKeys.length} key(s)</CardDescription>
        </CardHeader>
        <CardContent>
          {activeKeys.length === 0 ? (
            <p className="text-muted-foreground text-sm">No active API keys.</p>
          ) : (
            <div className="space-y-3">
              {activeKeys.map(key => (
                <div key={key.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="min-w-0 flex-1">
                    <div className="font-medium">{key.name}</div>
                    <div className="text-xs text-muted-foreground">
                      <code>{key.key_hash}</code>
                      {' · '}
                      {new Date(key.created_at * 1000).toLocaleDateString()}
                      {key.expires_at && (
                        <span className="ml-1 text-yellow-600">
                          · expires {new Date(key.expires_at * 1000).toLocaleDateString()}
                        </span>
                      )}
                      {key.scopes.length > 0 && key.scopes[0] !== '*' && (
                        <span className="ml-1">
                          · scopes: {key.scopes.join(', ')}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 ml-4">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleRotate(key.id)}
                      title="Rotate key"
                    >
                      <IconRefresh className="h-4 w-4" />
                    </Button>
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button size="sm" variant="ghost" title="Revoke key">
                          <IconTrash className="h-4 w-4 text-destructive" />
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Revoke &ldquo;{key.name}&rdquo;?</AlertDialogTitle>
                          <AlertDialogDescription>
                            Any applications using this key will stop working immediately.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => handleRevoke(key.id)}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                          >
                            {revokingId === key.id ? 'Revoking...' : 'Revoke'}
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
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
            <CardDescription>{revokedKeys.length} key(s)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {revokedKeys.map(key => (
                <div key={key.id} className="flex items-center justify-between p-3 border rounded-lg opacity-60">
                  <div>
                    <span className="font-medium">{key.name}</span>
                    <span className="ml-2 text-xs text-muted-foreground">
                      <code>{key.key_hash}</code>
                      {' · revoked'}
                    </span>
                  </div>
                  <Badge variant="secondary">Revoked</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </PageContainer>
  )
}
