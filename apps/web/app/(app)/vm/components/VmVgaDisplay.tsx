'use client'

// VGA 0xB8000 screen render — colored cell grid or plain text fallback, with a
// fullscreen toggle. Extracted from page.tsx by the page-decomposition refactor.

import { useState } from 'react'
import { Button, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'

export function VGADisplay({
  text,
  cells,
}: {
  text?: string
  cells?: { ch: string; fg: string; bg: string }[]
}) {
  const [fullScreen, setFullScreen] = useState(false)

  // Render colored cells as rows of spans
  const renderCells = () => {
    if (!cells) return null
    const rows: React.ReactNode[] = []
    for (let row = 0; row < 25; row++) {
      const rowCells = cells.slice(row * 80, (row + 1) * 80)
      // Skip empty rows
      const hasContent = rowCells.some((c) => c.ch !== ' ')
      if (!hasContent) continue
      rows.push(
        <div key={row}>
          {rowCells.map((c, col) => (
            <span key={col} style={{ color: c.fg, backgroundColor: c.bg }}>
              {c.ch}
            </span>
          ))}
        </div>,
      )
    }
    return rows
  }

  return (
    <Card className={fullScreen ? 'fixed inset-4 z-50' : ''}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Screen (VGA 0xB8000)</CardTitle>
          <Button size="sm" variant="ghost" onClick={() => setFullScreen(!fullScreen)}>
            {fullScreen ? 'Exit' : 'Fullscreen'}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div
          className={`bg-black font-mono text-xs p-3 rounded overflow-y-auto whitespace-pre ${
            fullScreen ? 'h-[calc(100dvh-8rem)]' : 'h-40'
          }`}
        >
          {cells ? (
            renderCells()
          ) : text ? (
            <span className="text-success">{text}</span>
          ) : (
            <span className="text-success">Programs that write to 0xB8000 will appear here.</span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
