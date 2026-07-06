'use client'

import { forwardRef, type LabelHTMLAttributes } from 'react'

const Label = forwardRef<HTMLLabelElement, LabelHTMLAttributes<HTMLLabelElement>>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className={[
        'text-sm font-medium leading-none text-foreground peer-disabled:cursor-not-allowed peer-disabled:opacity-70',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      {...props}
    />
  )
)
Label.displayName = 'Label'

export { Label }
