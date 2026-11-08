'use client'

// Execution trace table (debug runs). Extracted from page.tsx by the
// page-decomposition refactor — pure presentation.

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { VMRunResult } from '@/lib/vm-controller'

export function VmTrace({ trace }: { trace?: VMRunResult['trace'] }) {
  if (!trace || trace.length === 0) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Execution Trace (first {trace.length} steps)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-64 overflow-x-auto overflow-y-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-muted-foreground">
                <th className="text-left py-1 px-2">#</th>
                <th className="text-left py-1 px-2">EIP</th>
                <th className="text-left py-1 px-2">Opcode</th>
                <th className="text-left py-1 px-2">Operands</th>
              </tr>
            </thead>
            <tbody>
              {trace.map((t, i) => (
                <tr key={i} className="border-t border-border/30">
                  <td className="py-1 px-2">{t.step}</td>
                  <td className="py-1 px-2">{t.eip}</td>
                  <td className="py-1 px-2">{t.opcode}</td>
                  <td className="py-1 px-2">{t.operands}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
