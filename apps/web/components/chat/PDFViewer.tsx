'use client'

import { useState, useCallback } from 'react'
import { Button } from '@sloughgpt/strui'
import { Chip } from '@sloughgpt/strui'
import { IconChevronLeft, IconChevronRight, IconX } from '@sloughgpt/strui'

interface PDFViewerProps {
  pages: string[]
  filename: string
  defaultExpanded?: boolean
  onClose?: () => void
  suggestions?: string[]
  onSuggestionClick?: (text: string) => void
}

export function PDFViewer({ pages, filename, defaultExpanded = true, onClose, suggestions, onSuggestionClick }: PDFViewerProps) {
  const [currentPage, setCurrentPage] = useState(0)
  const [expanded, setExpanded] = useState(defaultExpanded)

  const totalPages = pages.length

  const goPrev = useCallback(() => {
    setCurrentPage(p => Math.max(0, p - 1))
  }, [])

  const goNext = useCallback(() => {
    setCurrentPage(p => Math.min(totalPages - 1, p + 1))
  }, [totalPages])

  if (!pages || pages.length === 0) return null

  return (
    <div className="border border-border rounded-lg overflow-hidden bg-card">
      {/* Header bar */}
      <div className="flex items-center justify-between px-3 py-2 bg-muted/50 border-b border-border">
        <div className="flex items-center gap-2 min-w-0">
          <svg className="h-4 w-4 shrink-0 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
          <span className="text-xs font-medium truncate">{filename}</span>
          <span className="text-xs text-muted-foreground shrink-0">({totalPages} {totalPages === 1 ? 'page' : 'pages'})</span>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setExpanded(e => !e)} aria-label={expanded ? 'Collapse' : 'Expand'}>
            {expanded ? (
              <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
            ) : (
              <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            )}
          </Button>
          {onClose && (
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose} aria-label="Remove PDF">
              <IconX className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>

      {/* Page content */}
      {expanded && (
        <div className="p-2">
          <div className="relative flex items-center justify-center bg-muted/30 rounded-lg min-h-[200px]">
            {totalPages > 1 && (
              <Button
                variant="ghost"
                size="icon"
                className="absolute left-1 z-10 h-8 w-8 rounded-full bg-background/80 hover:bg-background shadow-sm"
                onClick={goPrev}
                disabled={currentPage === 0}
                aria-label="Previous page"
              >
                <IconChevronLeft className="h-4 w-4" />
              </Button>
            )}
            <img
              src={pages[currentPage]}
              alt={`${filename} — page ${currentPage + 1}`}
              className="max-h-[250px] sm:max-h-[400px] w-auto object-contain rounded"
            />
            {totalPages > 1 && (
              <Button
                variant="ghost"
                size="icon"
                className="absolute right-1 z-10 h-8 w-8 rounded-full bg-background/80 hover:bg-background shadow-sm"
                onClick={goNext}
                disabled={currentPage === totalPages - 1}
                aria-label="Next page"
              >
                <IconChevronRight className="h-4 w-4" />
              </Button>
            )}
          </div>
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 mt-2">
              <span className="text-xs text-muted-foreground">
                Page {currentPage + 1} of {totalPages}
              </span>
            </div>
          )}
          {suggestions && suggestions.length > 0 && onSuggestionClick && (
            <div className="mt-3 pt-3 border-t border-border">
              <p className="text-xs text-muted-foreground mb-2">Try asking:</p>
              <div className="flex flex-wrap gap-1.5">
                {suggestions.map((s, i) => (
                  <Chip
                    key={i}
                    label={s}
                    onClick={() => onSuggestionClick(s)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
