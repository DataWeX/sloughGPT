'use client'

import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'

interface VMRegister {
  name: string
  value: number
  hex: string
}

interface VMResult {
  success: boolean
  exit_code: number
  steps_executed: number
  elapsed_ms: number
  status: string
  error?: string
  registers: VMRegister[]
  eip_hex: string
}

interface VMResultCardProps {
  result: VMResult | null
}

export function VMResultCard({ result }: VMResultCardProps) {
  if (!result) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Result</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Status</span>
          <Badge variant={result.success ? 'success' : 'error'} size="sm">
            {result.status}
          </Badge>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Exit code</span>
          <span className="font-mono">0x{result.exit_code.toString(16).toUpperCase()}</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Steps</span>
          <span className="font-mono">{result.steps_executed.toLocaleString()}</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Time</span>
          <span className="font-mono">{result.elapsed_ms.toFixed(1)}ms</span>
        </div>
        {result.error && (
          <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">{result.error}</div>
        )}
        {result.registers.length > 0 && (
          <div className="pt-2 border-t">
            <p className="text-xs text-muted-foreground mb-1">Registers</p>
            <div className="grid grid-cols-2 gap-1">
              {result.registers.map((reg) => (
                <div key={reg.name} className="flex justify-between text-xs font-mono px-2 py-1 bg-muted/30 rounded">
                  <span className="text-muted-foreground">{reg.name}</span>
                  <span>{reg.hex}</span>
                </div>
              ))}
            </div>
            <div className="flex justify-between text-xs font-mono px-2 py-1 bg-muted/30 rounded mt-1">
              <span className="text-muted-foreground">EIP</span>
              <span>{result.eip_hex}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
