'use client'

import * as DropdownMenuPrimitive from '@radix-ui/react-dropdown-menu'
import { forwardRef } from 'react'

const DropdownMenu = DropdownMenuPrimitive.Root
const DropdownMenuTrigger = DropdownMenuPrimitive.Trigger
const DropdownMenuPortal = DropdownMenuPrimitive.Portal
const DropdownMenuGroup = DropdownMenuPrimitive.Group
const DropdownMenuSub = DropdownMenuPrimitive.Sub
const DropdownMenuRadioGroup = DropdownMenuPrimitive.RadioGroup

const DropdownMenuContent = forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Content> & { align?: 'start' | 'center' | 'end'; sideOffset?: number }
>(({ className, align = 'end', sideOffset = 6, ...props }, ref) => (
  <DropdownMenuPrimitive.Portal>
    <DropdownMenuPrimitive.Content
      ref={ref}
      align={align}
      sideOffset={sideOffset}
      className={[
        'z-50 min-w-[10rem] bg-popover border border-border rounded-lg shadow-xl p-1',
        'data-[state=open]:animate-in data-[state=closed]:animate-out',
        'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
        'data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95',
        'data-[side=bottom]:slide-in-from-top-2',
        'data-[side=top]:slide-in-from-bottom-2',
        className,
      ].filter(Boolean).join(' ')}
      {...props}
    />
  </DropdownMenuPrimitive.Portal>
))
DropdownMenuContent.displayName = 'DropdownMenuContent'

const DropdownMenuItem = forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Item> & { destructive?: boolean; inset?: boolean }
>(({ className, destructive, inset, ...props }, ref) => (
  <DropdownMenuPrimitive.Item
    ref={ref}
    className={[
      'relative flex items-center gap-2 px-2.5 py-2 text-sm rounded-md cursor-pointer outline-none transition-colors duration-150',
      'focus:bg-primary/10',
      'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
      destructive ? 'text-destructive focus:text-destructive' : 'text-foreground',
      inset && 'pl-8',
      className,
    ].filter(Boolean).join(' ')}
    {...props}
  />
))
DropdownMenuItem.displayName = 'DropdownMenuItem'

const DropdownMenuCheckboxItem = forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.CheckboxItem>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.CheckboxItem>
>(({ className, children, ...props }, ref) => (
  <DropdownMenuPrimitive.CheckboxItem
    ref={ref}
    className={[
      'relative flex items-center gap-2 px-2.5 py-2 text-sm rounded-md cursor-pointer outline-none transition-colors duration-150',
      'focus:bg-primary/10',
      'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
      className,
    ].filter(Boolean).join(' ')}
    {...props}
  >
    {children}
    <DropdownMenuPrimitive.ItemIndicator className="ml-auto">
      <svg className="h-4 w-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    </DropdownMenuPrimitive.ItemIndicator>
  </DropdownMenuPrimitive.CheckboxItem>
))
DropdownMenuCheckboxItem.displayName = 'DropdownMenuCheckboxItem'

const DropdownMenuRadioItem = forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.RadioItem>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.RadioItem>
>(({ className, children, ...props }, ref) => (
  <DropdownMenuPrimitive.RadioItem
    ref={ref}
    className={[
      'relative flex items-center gap-2 px-2.5 py-2 text-sm rounded-md cursor-pointer outline-none transition-colors duration-150',
      'focus:bg-primary/10',
      'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
      className,
    ].filter(Boolean).join(' ')}
    {...props}
  >
    {children}
    <DropdownMenuPrimitive.ItemIndicator className="ml-auto">
      <svg className="h-3 w-3 text-primary" fill="currentColor" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="6" />
      </svg>
    </DropdownMenuPrimitive.ItemIndicator>
  </DropdownMenuPrimitive.RadioItem>
))
DropdownMenuRadioItem.displayName = 'DropdownMenuRadioItem'

const DropdownMenuLabel = forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Label> & { inset?: boolean }
>(({ className, inset, ...props }, ref) => (
  <DropdownMenuPrimitive.Label
    ref={ref}
    className={['px-2.5 py-2 text-xs font-semibold text-muted-foreground', inset && 'pl-8', className].filter(Boolean).join(' ')}
    {...props}
  />
))
DropdownMenuLabel.displayName = 'DropdownMenuLabel'

const DropdownMenuSeparator = forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.Separator ref={ref} className={['my-1 h-px bg-border', className].filter(Boolean).join(' ')} {...props} />
))
DropdownMenuSeparator.displayName = 'DropdownMenuSeparator'

const DropdownMenuSubTrigger = forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.SubTrigger>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.SubTrigger> & { inset?: boolean }
>(({ className, inset, children, ...props }, ref) => (
  <DropdownMenuPrimitive.SubTrigger
    ref={ref}
    className={[
      'relative flex items-center gap-2 px-2.5 py-2 text-sm rounded-md cursor-pointer outline-none transition-colors duration-150',
      'focus:bg-primary/10 data-[state=open]:bg-primary/10',
      inset && 'pl-8',
      className,
    ].filter(Boolean).join(' ')}
    {...props}
  >
    {children}
    <svg className="ml-auto h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </svg>
  </DropdownMenuPrimitive.SubTrigger>
))
DropdownMenuSubTrigger.displayName = 'DropdownMenuSubTrigger'

const DropdownMenuSubContent = forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.SubContent>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.SubContent>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.SubContent
    ref={ref}
    className={[
      'z-50 min-w-[10rem] bg-popover border border-border rounded-lg shadow-xl p-1',
      'data-[state=open]:animate-in data-[state=closed]:animate-out',
      'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
      'data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95',
      className,
    ].filter(Boolean).join(' ')}
    {...props}
  />
))
DropdownMenuSubContent.displayName = 'DropdownMenuSubContent'

export {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem,
  DropdownMenuCheckboxItem, DropdownMenuRadioItem, DropdownMenuLabel,
  DropdownMenuSeparator, DropdownMenuSubTrigger, DropdownMenuSubContent,
  DropdownMenuGroup, DropdownMenuPortal, DropdownMenuSub, DropdownMenuRadioGroup,
}
