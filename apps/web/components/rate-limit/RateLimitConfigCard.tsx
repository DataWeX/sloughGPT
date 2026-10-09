'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'

interface RateLimitConfigCardProps {
  config?: Record<string, unknown> | null
  loading?: boolean
}

export function RateLimitConfigCard({
  config,
  loading = false,
}: RateLimitConfigCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Configuration</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5">
        <pre className="rounded-lg bg-muted/20 p-2.5 text-[11px] font-mono overflow-auto whitespace-pre-wrap">
          {config ? JSON.stringify(config, null, 2) : 'Loading...'}
        </pre>
      </CardContent>
    </Card>
  )
}
