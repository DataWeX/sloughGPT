'use client'

import * as DropdownMenuPrimitive from '@radix-ui/react-dropdown-menu'

import { Button } from '@/components/ui/button'

interface SelectOption {
  value: string
  label: string
}

interface SelectProps {
  value: string
  onValueChange: (value: string) => void
  options: SelectOption[]
  placeholder?: string
  className?: string
}

export function Select({ value, onValueChange, options, placeholder = 'Select...', className = '' }: SelectProps) {
  const selected = options.find(o => o.value === value)

  return (
    <DropdownMenuPrimitive.Root>
      <DropdownMenuPrimitive.Trigger asChild>
        <Button variant="outline" size="sm" className={className}>
          <span className={selected ? 'text-foreground' : 'text-muted-foreground'}>
            {selected?.label || placeholder}
          </span>
          <svg className="w-4 h-4 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </Button>
      </DropdownMenuPrimitive.Trigger>

      <DropdownMenuPrimitive.Portal>
        <DropdownMenuPrimitive.Content
          align="start"
          className="z-50 w-[var(--radix-dropdown-menu-trigger-width)] overflow-hidden rounded-lg border bg-background p-1 shadow-lg animate-in fade-in-0 zoom-in-95"
        >
          {options.map((option, index) => (
            <DropdownMenuPrimitive.Item
              key={option.value}
              onSelect={() => onValueChange(option.value)}
              className={`relative flex cursor-pointer select-none items-center rounded px-3 py-2 text-sm outline-none transition-colors ${
                value === option.value
                  ? 'bg-primary/10 text-primary font-medium'
                  : 'sl-hover-primary text-foreground hover:text-primary'
              } ${index === 0 ? 'rounded-t-md' : ''} ${index === options.length - 1 ? 'rounded-b-md' : ''}`}
            >
              {value === option.value && (
                <svg className="mr-2 h-4 w-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              )}
              {option.label}
            </DropdownMenuPrimitive.Item>
          ))}
        </DropdownMenuPrimitive.Content>
      </DropdownMenuPrimitive.Portal>
    </DropdownMenuPrimitive.Root>
  )
}
