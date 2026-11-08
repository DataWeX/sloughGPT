'use client'

// x86 assembly quick-reference panel (toggled by the Ref button). Extracted
// from page.tsx by the page-decomposition refactor — pure presentation.

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'

export function VmReference({ open }: { open: boolean }) {
  if (!open) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">x86 Reference</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div>
            <p className="font-medium mb-1">Data Movement</p>
            <pre className="text-muted-foreground">
              {'MOV dst, src\nPUSH val\nPOP dst\nXCHG a, b\nLEA dst, [addr]'}
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1">Arithmetic</p>
            <pre className="text-muted-foreground">
              {'ADD dst, src\nSUB dst, src\nINC reg\nDEC reg\nMUL src\nDIV src\nNEG dst'}
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1">Logic / Shift</p>
            <pre className="text-muted-foreground">
              {
                'AND dst, src\nOR  dst, src\nXOR dst, src\nNOT dst\nSHL dst, n\nSHR dst, n\nCMP a, b\nTEST a, b'
              }
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1">Control Flow</p>
            <pre className="text-muted-foreground">
              {'JMP label\nJE / JNE label\nJG / JL label\nJGE / JLE label\nCALL func\nRET\nHLT'}
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1">String</p>
            <pre className="text-muted-foreground">
              {'LODSB/W/D\nSTOSB/W/D\nMOVSB/W/D\nCMPSB/W/D\nSCASB/W/D\nREP prefix'}
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1">Stack</p>
            <pre className="text-muted-foreground">
              {'PUSHAD\nPOPAD\nPUSHFD\nPOPFD\nENTER\nLEAVE'}
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1">Interrupts</p>
            <pre className="text-muted-foreground">
              {
                'INT 0x80\n  EAX=4 write\n  EAX=1 exit\n  EAX=3 read\n  EAX=28 train start\n  EAX=29 train status\n  EAX=30 train result\nINT 0x10 video\nINT 0x16 keyboard'
              }
            </pre>
          </div>
          <div>
            <p className="font-medium mb-1">Registers</p>
            <pre className="text-muted-foreground">
              {
                'EAX  accumulator\nECX  counter\nEDX  data\nEBX  base\nESP  stack ptr\nEBP  base ptr\nESI  src index\nEDI  dst index'
              }
            </pre>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
