'use client'

import { useEffect, useRef, useState } from 'react'
import { trainingController } from '@/lib/controllers'

export function TrainingLogViewer() {
  const [lines, setLines] = useState<string[]>([])
  const [expanded, setExpanded] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!expanded) return
    let cancelled = false
    const fetchLines = async () => {
      try {
        const lns = await trainingController.getTrainingLog()
        if (!cancelled) setLines(lns)
      } catch { /* ignore */ }
    }
    void fetchLines()
    const id = setInterval(fetchLines, 5000)
    return () => { cancelled = true; clearInterval(id) }
  }, [expanded])

  useEffect(() => {
    if (expanded) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines, expanded])

  return (
    <div className="space-y-2">
      <button
        className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? 'Hide' : 'Show'} recent log entries
        {lines.length > 0 && !expanded && (
          <span className="ml-1 text-[11px] text-muted-foreground/60">({lines.length} lines)</span>
        )}
      </button>
      {expanded && (
        <div className="max-h-48 overflow-y-auto rounded border border-border/50 bg-black/5 dark:bg-white/5 p-2 font-mono text-[11px] leading-relaxed">
          {lines.length === 0 ? (
            <p className="text-muted-foreground/60 italic">No log entries yet</p>
          ) : (
            lines.map((line, i) => (
              <div key={i} className="text-muted-foreground hover:text-foreground transition-colors">
                {line}
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  )
}
