'use client'

import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'

export interface RateLimitCheck {
  allowed: boolean
  wait_time: number
}

interface RateLimitCheckCardProps {
  checkResult?: RateLimitCheck | null
  onCheck?: () => Promise<void>
  checking?: boolean
}

export function RateLimitCheckCard({
  checkResult,
  onCheck,
  checking = false,
}: RateLimitCheckCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Check Rate Limit</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 px-2.5 pb-2.5">
        <p className="text-[10px] text-muted-foreground/60">
          Simulates a request check to see if the current IP would be rate limited.
        </p>
        <Button size="sm" className="h-7 text-[11px]" onClick={onCheck} disabled={checking}>
          {checking ? 'Checking...' : 'Check Now'}
        </Button>
        {checkResult && (
          <div className="rounded-lg bg-muted/20 px-2.5 py-2 text-[11px] space-y-0.5">
            <div className="flex justify-between">
              <span className="font-medium">
                Allowed: {checkResult.allowed ? 'Yes' : 'No'}
              </span>
              {!checkResult.allowed && (
                <span className="text-warning font-mono tabular-nums">
                  Wait: {checkResult.wait_time.toFixed(1)}s
                </span>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
