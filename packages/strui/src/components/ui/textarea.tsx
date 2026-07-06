'use client'

import { forwardRef, type TextareaHTMLAttributes } from 'react'

export type TextareaProps = TextareaHTMLAttributes<HTMLTextAreaElement>

const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(({ className, ...props }, ref) => {
  return (
    <textarea
      className={[
        'flex w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground shadow-sm',
        'transition-[border-color,box-shadow,background-color] duration-200',
        'placeholder:text-muted-foreground selection:bg-primary/20 selection:text-foreground',
        'hover:border-primary/50',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background focus-visible:border-primary/60',
        'disabled:cursor-not-allowed disabled:opacity-50',
        'min-h-20 resize-y',
        className,
      ].filter(Boolean).join(' ')}
      ref={ref}
      {...props}
    />
  )
})
Textarea.displayName = 'Textarea'

export { Textarea }
