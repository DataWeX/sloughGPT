import { cn } from '../../lib/cn'

export interface SourceItem {
  title: string
  url?: string
  snippet?: string
}

export interface SourceListProps {
  sources: SourceItem[]
  className?: string
}

/** Numbered references block for RAG answers (below assistant message). */
export function SourceList({ sources, className }: SourceListProps) {
  if (sources.length === 0) return null

  return (
    <aside className={cn('rounded-none border border-border bg-muted/20 p-3 text-sm', className)} aria-label="Sources">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sources</h3>
      <ol className="list-decimal space-y-2 pl-5 text-muted-foreground">
        {sources.map((s, i) => (
          <li key={i} className="leading-snug">
            {s.url ? (
              <a
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-primary hover:underline"
              >
                {s.title}
              </a>
            ) : (
              <span className="font-medium text-foreground">{s.title}</span>
            )}
            {s.snippet ? <span className="mt-1 block text-xs opacity-90">{s.snippet}</span> : null}
          </li>
        ))}
      </ol>
    </aside>
  )
}
