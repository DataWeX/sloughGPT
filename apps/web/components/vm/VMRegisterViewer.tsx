'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

interface VMRegister {
  name: string
  value: number
  hex: string
}

interface VMRegisterViewerProps {
  registers: VMRegister[]
  eipHex: string
  onCopyAll?: (text: string) => void
  onCopyRegister?: (hex: string) => void
}

export function VMRegisterViewer({
  registers,
  eipHex,
  onCopyAll,
  onCopyRegister,
}: VMRegisterViewerProps) {
  if (registers.length === 0) return null

  const handleCopyAll = () => {
    const text = registers.map((r) => `${r.name} = ${r.hex}`).join('\n')
    onCopyAll?.(text)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Registers</CardTitle>
          <Button size="sm" variant="ghost" onClick={handleCopyAll}>
            Copy
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-1">
          {registers.map((reg) => (
            <button
              key={reg.name}
              className="flex justify-between text-xs font-mono px-2 py-1 bg-muted/30 rounded hover:bg-muted/60 text-left transition-colors"
              onClick={() => onCopyRegister?.(reg.hex)}
              title="Click to copy"
            >
              <span className="text-muted-foreground">{reg.name}</span>
              <span>{reg.hex}</span>
            </button>
          ))}
        </div>
        <div className="flex justify-between text-xs font-mono px-2 py-1 bg-muted/30 rounded mt-1">
          <span className="text-muted-foreground">EIP</span>
          <span>{eipHex}</span>
        </div>
      </CardContent>
    </Card>
  )
}
