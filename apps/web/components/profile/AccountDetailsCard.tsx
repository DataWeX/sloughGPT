'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface AccountDetailsCardProps {
  userId?: string
  tenantId?: string
  lastLoginAt?: string
  createdAt?: string
}

export function AccountDetailsCard({
  userId = '—',
  tenantId = '—',
  lastLoginAt,
  createdAt,
}: AccountDetailsCardProps) {
  const formatDate = (d?: string) => (d ? new Date(d).toLocaleString() : '—')

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Account Details</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div>
            <span className="text-muted-foreground">User ID: </span>
            <span className="font-mono">{userId}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Tenant ID: </span>
            <span className="font-mono">{tenantId}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Last Login: </span>
            <span>{formatDate(lastLoginAt)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Created: </span>
            <span>{formatDate(createdAt)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
