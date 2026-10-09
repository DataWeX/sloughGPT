'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { Zap } from 'lucide-react'

export interface BulkAction {
  label: string
  onClick: () => void
}

export interface ShortcutBulkActionsCardProps {
  actions: BulkAction[]
  disabled?: boolean
}

export function ShortcutBulkActionsCard({ actions, disabled = false }: ShortcutBulkActionsCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs flex items-center gap-2">
          <Zap className="h-3.5 w-3.5" />
          Quick Actions
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-1.5">
          {actions.map((action, i) => (
            <Button
              key={i}
              size="sm"
              variant="outline"
              className="h-6 text-[10px]"
              onClick={action.onClick}
              disabled={disabled}
            >
              {action.label}
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
