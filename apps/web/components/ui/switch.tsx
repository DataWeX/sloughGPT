'use client'

import * as SwitchPrimitive from '@radix-ui/react-switch'
import { forwardRef } from 'react'

const Switch = forwardRef<HTMLButtonElement, SwitchPrimitive.SwitchProps>(
  ({ className, ...props }, ref) => (
    <SwitchPrimitive.Root
      ref={ref}
      className={[
        'peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors duration-200',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
        'disabled:cursor-not-allowed disabled:opacity-50',
        'data-[state=checked]:bg-primary data-[state=unchecked]:bg-muted',
        className,
      ].filter(Boolean).join(' ')}
      {...props}
    >
      <SwitchPrimitive.Thumb
        className={[
          'pointer-events-none block h-5 w-5 rounded-full bg-card shadow-md ring-0 transition-transform duration-200',
          'data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0',
        ].join(' ')}
      />
    </SwitchPrimitive.Root>
  )
)
Switch.displayName = 'Switch'

export { Switch }
