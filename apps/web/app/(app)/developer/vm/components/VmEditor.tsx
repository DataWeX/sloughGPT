'use client'

// Assembly source editor card: Save/Load/Share actions, line-number gutter,
// and the code textarea. Extracted from page.tsx by the page-decomposition
// refactor — pure presentation over hook state.

import { type Dispatch, type RefObject, type SetStateAction } from 'react'
import { Button, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { COPY_FEEDBACK_DURATION_MS } from '@/lib/constants'

interface VmEditorProps {
  source: string
  setSource: Dispatch<SetStateAction<string>>
  textareaRef: RefObject<HTMLTextAreaElement | null>
  handleKeyDown: (e: React.KeyboardEvent) => void
  copied: boolean
  setCopied: Dispatch<SetStateAction<boolean>>
  copiedTimerRef: RefObject<ReturnType<typeof setTimeout> | null>
}

export function VmEditor({
  source,
  setSource,
  textareaRef,
  handleKeyDown,
  copied,
  setCopied,
  copiedTimerRef,
}: VmEditorProps) {
  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Assembly Source</CardTitle>
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                const blob = new Blob([source], { type: 'text/plain' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = 'program.asm'
                a.click()
                URL.revokeObjectURL(url)
              }}
            >
              Save
            </Button>
            <label className="cursor-pointer">
              <input
                type="file"
                accept=".asm,.txt"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (!file) return
                  const reader = new FileReader()
                  reader.onload = () => setSource(reader.result as string)
                  reader.readAsText(file)
                }}
              />
              <span className="inline-flex items-center justify-center h-7 px-3 text-xs font-medium rounded-md border border-border hover:bg-muted/50 transition-colors">
                Load
              </span>
            </label>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                const encoded = btoa(source)
                const url = `${window.location.origin}${window.location.pathname}#code=${encoded}`
                navigator.clipboard.writeText(url)
                setCopied(true)
                if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
                copiedTimerRef.current = setTimeout(
                  () => setCopied(false),
                  COPY_FEEDBACK_DURATION_MS,
                )
              }}
            >
              {copied ? 'Copied!' : 'Share'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative border rounded-md overflow-hidden">
          <div className="flex h-80">
            {/* Line numbers */}
            <div className="select-none text-right text-xs text-muted-foreground font-mono bg-muted/20 border-r border-border/40 py-3 px-2 overflow-hidden">
              {source.split('\n').map((_, i) => (
                <div key={i} className="leading-5">
                  {i + 1}
                </div>
              ))}
            </div>
            {/* Editor */}
            <textarea
              ref={textareaRef}
              value={source}
              onChange={(e) => setSource(e.target.value)}
              onKeyDown={handleKeyDown}
              aria-label="Assembly source code"
              className="flex-1 h-full p-3 font-mono text-sm bg-background resize-none focus:outline-none leading-5"
              spellCheck={false}
              placeholder="[BITS 32]&#10;[ORG 0x1000]&#10;&#10;MOV EAX, 42&#10;HLT"
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-1">Ctrl+Enter to run</p>
      </CardContent>
    </Card>
  )
}
