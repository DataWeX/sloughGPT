'use client'

import { Card, CardHeader, CardTitle, CardContent, Badge, Button } from '@sloughgpt/strui'
import { IconRefresh, IconTrash } from '@/components/icons/NavIcons'

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
  title?: string
  onRotate?: (keyId: string) => void
  onRevoke?: (keyId: string) => void
}

export function ApiKeyListCard({
  keys,
  title = 'Active Keys',
  onRotate,
  onRevoke,
}: ApiKeyListCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {keys.length === 0 ? (
          <p className="text-muted-foreground text-sm">No API keys.</p>
        ) : (
          <div className="space-y-3">
            {keys.map((key) => (
              <div
                key={key.id}
                className={`flex items-center justify-between p-3 border rounded-lg ${key.revoked ? 'opacity-60' : ''}`}
              >
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
                  </div>
                </div>
                <div className="flex items-center gap-1 ml-4">
                  {key.revoked ? (
                    <Badge variant="secondary">Revoked</Badge>
                  ) : (
                    <>
                      {onRotate && (
                        <Button size="sm" variant="ghost" onClick={() => onRotate(key.id)} title="Rotate key">
                          <IconRefresh className="h-4 w-4" />
                        </Button>
                      )}
                      {onRevoke && (
                        <Button size="sm" variant="ghost" onClick={() => onRevoke(key.id)} title="Revoke key">
                          <IconTrash className="h-4 w-4 text-destructive" />
                        </Button>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
