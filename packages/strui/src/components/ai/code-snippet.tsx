import * as React from 'react'

import { cn } from '../../lib/cn'

export interface CodeSnippetProps extends React.HTMLAttributes<HTMLPreElement> {
  /** When true, allows horizontal scroll on small screens without breaking layout. */
  scroll?: boolean
}

/** Model output / fenced code — monospace, lattice border. */
export function CodeSnippet({ className, scroll = true, children, ...props }: CodeSnippetProps) {
  return (
    <pre
      className={cn(
        'sl-code block p-3 text-left text-xs sm:text-sm',
        scroll && 'max-w-full overflow-x-auto str-chat-scroll',
        className,
      )}
      {...props}
    >
      <code>{children}</code>
    </pre>
  )
}
