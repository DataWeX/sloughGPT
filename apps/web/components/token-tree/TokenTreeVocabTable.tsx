'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

export interface VocabEntry {
  id: number
  token: string
  freq: number
  is_special: boolean
  is_merged: boolean
}

export interface TokenTreeVocabTableProps {
  entries: VocabEntry[]
  total: number
  offset: number
  onPageChange: (offset: number) => void
}

export function TokenTreeVocabTable({ entries, total, offset, onPageChange }: TokenTreeVocabTableProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Vocabulary ({total} tokens)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 px-2.5 pb-2.5">
        <div className="max-h-[400px] overflow-y-auto">
          <table className="w-full text-[11px]" aria-label="Vocabulary table">
            <thead>
              <tr className="border-b border-border/30 text-left text-[10px] text-muted-foreground/60">
                <th scope="col" className="pb-1 font-medium">ID</th>
                <th scope="col" className="pb-1 font-medium">Token</th>
                <th scope="col" className="pb-1 font-medium">Freq</th>
                <th scope="col" className="pb-1 font-medium">Type</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(e => (
                <tr key={e.id} className="border-b border-border/20">
                  <td className="py-1 font-mono text-muted-foreground/60 tabular-nums">{e.id}</td>
                  <td className="py-1 font-mono">{e.token}</td>
                  <td className="py-1 font-mono tabular-nums">{e.freq}</td>
                  <td className="py-1">
                    {e.is_special && <span className="rounded-full bg-warning/10 px-1.5 py-0.5 text-[9px] text-warning">special</span>}
                    {e.is_merged && <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[9px] text-primary">merged</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex justify-center gap-1.5">
          <Button size="sm" variant="outline" className="h-6 text-[10px]" disabled={offset === 0} onClick={() => onPageChange(offset - 50)}>Previous</Button>
          <span className="text-[10px] text-muted-foreground/60 self-center font-mono tabular-nums">{offset + 1}-{Math.min(offset + 50, total)} of {total}</span>
          <Button size="sm" variant="outline" className="h-6 text-[10px]" disabled={offset + 50 >= total} onClick={() => onPageChange(offset + 50)}>Next</Button>
        </div>
      </CardContent>
    </Card>
  )
}
