'use client'

/** Files tab — file stats and browsing. */
import { FileStatsCard } from '@/components/files/FileStatsCard'
import { useFileList } from '@/lib/cache'
import { IconRefresh, Skeleton, cn } from '@sloughgpt/strui'
import { useState } from 'react'

export function FilesTab() {
  const { data: files = [], isLoading: loading, refetch } = useFileList()
  const [search, setSearch] = useState('')

  const filtered = files.filter((f) => f.filename.toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="space-y-4">
      <FileStatsCard files={files} />
      <div className="rounded-xl border border-border/20 bg-card overflow-hidden">
        <div className="flex items-center justify-between h-11 px-4 bg-muted/30 border-b border-border/20">
          <span className="text-[11px] font-medium text-muted-foreground/80">Documents</span>
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Search..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-40 h-7 rounded-md border border-border/20 bg-muted/20 px-2.5 text-[11px] text-muted-foreground placeholder:text-muted-foreground/40 outline-none focus:border-white/[0.12] transition-colors font-mono"
              aria-label="Search files"
            />
            <button
              type="button"
              onClick={() => {
                refetch()
              }}
              className="h-7 w-7 flex items-center justify-center rounded-md border border-border/20 bg-muted/20 text-muted-foreground/80 hover:text-muted-foreground hover:border-white/[0.12] transition-colors"
              aria-label="Refresh files"
            >
              <IconRefresh className="w-3 h-3" />
            </button>
          </div>
        </div>
        <div className="px-4 py-2">
          {loading ? (
            <div className="space-y-2 py-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-7 w-full bg-muted/30" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <p className="text-[11px] text-muted-foreground/60 py-6 text-center">
              {search ? 'No files match your search.' : 'No files uploaded yet.'}
            </p>
          ) : (
            <div className="max-h-72 overflow-y-auto py-1">
              {filtered.map((f, i) => (
                <div
                  key={f.id}
                  className={cn(
                    'flex items-center justify-between px-2.5 py-2 rounded-lg text-[11px] transition-colors hover:bg-white/[0.04]',
                    i > 0 && 'border-t border-white/[0.04]',
                  )}
                >
                  <span className="font-mono text-muted-foreground truncate">{f.filename}</span>
                  <span className="text-muted-foreground/60 shrink-0 ml-3 font-mono">
                    {f.size ? `${(f.size / 1024).toFixed(1)} KB` : '—'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
