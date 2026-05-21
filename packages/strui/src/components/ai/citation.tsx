import type { AnchorHTMLAttributes, HTMLAttributes, ReactNode } from 'react'

import { cn } from '../../lib/cn'

const markClass =
  'ml-0.5 inline align-super text-[0.65rem] font-semibold text-primary underline-offset-2'

export interface CitationProps {
  index: number
  children?: ReactNode
  className?: string
  href?: string
  title?: string
}

/** Inline RAG / source reference — link when `href` is set. */
export function Citation({ index, children, className, href, title, ...rest }: CitationProps) {
  const label = children ?? index
  const inner = <>[{label}]</>
  const t = title ?? `Source ${index}`

  if (href) {
    return (
      <a
        href={href}
        title={t}
        aria-label={t}
        className={cn(markClass, 'hover:underline', className)}
        {...(rest as AnchorHTMLAttributes<HTMLAnchorElement>)}
      >
        {inner}
      </a>
    )
  }

  return (
    <span
      title={t}
      aria-label={t}
      role="doc-noteref"
      className={cn(markClass, 'cursor-default no-underline', className)}
      {...(rest as HTMLAttributes<HTMLSpanElement>)}
    >
      {inner}
    </span>
  )
}
