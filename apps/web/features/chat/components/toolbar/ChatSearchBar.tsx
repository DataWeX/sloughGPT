'use client'

import React from 'react'
import { IconSearch, IconX, IconChevronDown, IconChevronUp } from '@sloughgpt/strui'
import { useChatToolbarContext } from '@/features/chat/contexts/ChatToolbarContext'

export function ChatSearchBar() {
  const ctx = useChatToolbarContext()
  const { query: searchQuery, onChange: onSearchChange, onClear, matchIndex, matchCount, onPrevMatch, onNextMatch, searchInputRef } = ctx.search
  return (
    <div className="relative w-36 sm:w-44">
      <IconSearch className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground/60 pointer-events-none" aria-hidden />
      <input
        ref={searchInputRef as React.Ref<HTMLInputElement> | undefined}
        type="text"
        placeholder="Search..."
        value={searchQuery}
        onChange={(e) => onSearchChange(e.target.value)}
        className="w-full pl-7 pr-6 py-1.5 text-xs rounded-lg border border-input bg-background/80 focus:outline-none focus:ring-1 focus:ring-primary/40 placeholder:text-muted-foreground/50 transition-shadow hover:shadow-sm"
        aria-label="Search messages"
      />
      {searchQuery && (
        <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
          {matchCount > 0 && (
            <span className="hidden sm:inline text-[10px] text-muted-foreground whitespace-nowrap mr-0.5" aria-live="polite" role="status" aria-atomic="true">
              {matchIndex + 1}/{matchCount}
            </span>
          )}
          <button
            type="button"
            onClick={onPrevMatch}
            className="text-muted-foreground hover:text-foreground disabled:opacity-30 h-6 w-6 sm:h-7 sm:w-7 flex items-center justify-center rounded"
            disabled={matchCount === 0}
            aria-label="Previous match"
          >
            <IconChevronUp className="h-3 w-3" aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={onNextMatch}
            className="text-muted-foreground hover:text-foreground disabled:opacity-30 h-6 w-6 sm:h-7 sm:w-7 flex items-center justify-center rounded"
            disabled={matchCount === 0}
            aria-label="Next match"
          >
            <IconChevronDown className="h-3 w-3" aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={onClear}
            className="text-muted-foreground hover:text-foreground transition-colors h-6 w-6 sm:h-7 sm:w-7 flex items-center justify-center rounded"
            aria-label="Clear search"
          >
            <IconX className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  )
}
