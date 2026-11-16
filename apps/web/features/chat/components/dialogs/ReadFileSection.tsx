'use client'

import { useState } from 'react'

interface ReadFileSectionProps {
  readLoading: boolean
  readFileData: { text: string; filename: string; pages: number } | null
  onFileSelected: (file: File) => void
  onRemove: () => void
}

const FORMATS = 'PDF, Word, TXT, MD, CSV, JSON'

export default function ReadFileSection({
  readLoading,
  readFileData,
  onFileSelected,
  onRemove,
}: ReadFileSectionProps) {
  const [dragging, setDragging] = useState(false)

  if (!readFileData) {
    // Idle = a slim one-line bar (~50px) — the attach flow is always mounted,
    // so it must not tax the transcript column. The full dropzone only expands
    // while a file is being dragged over or read.
    const expanded = dragging || readLoading

    return (
      <div
        className="px-3 py-2 border-b border-border/10 bg-muted/5"
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={(e) => {
          // Only collapse when the SECTION itself is left: child-to-child
          // moves bubble here with the child as target and must be ignored
          // (relatedTarget is unreliable across engines, so key on target).
          if (e.target !== e.currentTarget) return
          setDragging(false)
        }}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          const f = e.dataTransfer.files[0]
          if (f) onFileSelected(f)
        }}
      >
        <input
          type="file"
          accept=".pdf,.docx,.txt,.md,.csv,.json"
          className="sr-only peer"
          id="read-file-input"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onFileSelected(f)
          }}
        />
        {expanded ? (
          <div className="flex flex-col items-center gap-2 py-6 border-2 border-dashed border-border/30 rounded-lg text-center">
            <label
              htmlFor="read-file-input"
              className="cursor-pointer flex flex-col items-center gap-1 rounded peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring"
            >
              <span className="text-2xl">📄</span>
              <span className="text-sm font-medium">
                {readLoading ? 'Reading your file...' : 'Release to upload'}
              </span>
              <span className="text-[11px] text-muted-foreground">{FORMATS}</span>
            </label>
          </div>
        ) : (
          <label
            htmlFor="read-file-input"
            className="flex items-center gap-2 w-full px-3 py-1.5 border border-dashed border-border/30 rounded-lg text-sm cursor-pointer hover:border-primary/40 hover:bg-muted/10 transition-colors peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring"
          >
            <span className="text-base">📄</span>
            <span className="font-medium">Drop a file here or click to upload</span>
            <span className="hidden sm:inline text-[11px] text-muted-foreground ml-auto">
              {FORMATS}
            </span>
          </label>
        )}
      </div>
    )
  }

  return (
    <div className="px-3 py-2 border-b border-border/10 bg-muted/5">
      <div className="flex items-center gap-2 text-sm">
        <span className="text-base">📄</span>
        <span className="font-medium truncate max-w-[200px]">{readFileData.filename}</span>
        {readFileData.pages > 1 && (
          <span className="text-xs text-muted-foreground">({readFileData.pages} pages)</span>
        )}
        <button
          type="button"
          onClick={onRemove}
          className="active:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset ml-auto text-xs text-muted-foreground hover:text-foreground px-2 py-0.5 rounded hover:bg-muted/10"
        >
          Remove
        </button>
      </div>
    </div>
  )
}
