'use client'

import * as DialogPrimitive from '@radix-ui/react-dialog'
import { forwardRef } from 'react'

const Sheet = DialogPrimitive.Root
const SheetTrigger = DialogPrimitive.Trigger
const SheetClose = DialogPrimitive.Close
const SheetPortal = DialogPrimitive.Portal

const SheetOverlay = forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={[
      'fixed inset-0 z-50 bg-foreground/20 backdrop-blur-sm',
      'data-[state=open]:animate-in data-[state=closed]:animate-out',
      'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
      'duration-200',
      className,
    ].filter(Boolean).join(' ')}
    {...props}
  />
))
SheetOverlay.displayName = DialogPrimitive.Overlay.displayName

const SheetContent = forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & { side?: 'left' | 'right' }
>(({ className, children, side = 'left', ...props }, ref) => (
  <SheetPortal>
    <SheetOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={[
        'fixed top-0 z-50 h-dvh w-72 gap-0 border-r border-border bg-background p-0 shadow-xl',
        'duration-200',
        'data-[state=open]:animate-in data-[state=closed]:animate-out',
        side === 'left'
          ? 'left-0 data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left'
          : 'right-0 data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right',
        'max-h-dvh overflow-hidden',
        className,
      ].filter(Boolean).join(' ')}
      {...props}
    >
      {children}
    </DialogPrimitive.Content>
  </SheetPortal>
))
SheetContent.displayName = DialogPrimitive.Content.displayName

export { Sheet, SheetTrigger, SheetClose, SheetContent, SheetPortal }
