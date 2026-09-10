'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

export interface MemberRolesCardProps {
  roles: Record<string, number>
}

export function MemberRolesCard({ roles }: MemberRolesCardProps) {
  const entries = Object.entries(roles)

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Member Roles</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          {entries.map(([role, count]) => (
            <div key={role} className="flex items-center justify-between px-2 py-1.5 rounded text-[10px] hover:bg-muted/50">
              <span className="font-medium capitalize">{role}</span>
              <span className="text-muted-foreground">{count}</span>
            </div>
          ))}
          {entries.length === 0 && (
            <p className="text-[10px] text-muted-foreground text-center py-2">No members</p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
