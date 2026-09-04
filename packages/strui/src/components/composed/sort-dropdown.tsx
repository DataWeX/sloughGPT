'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { Button } from '../ui/button'
import { IconChevronDown } from '../ui/icons'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'

export interface SortOption<T extends string> {
  /** The sort value */
  value: T
  /** Display label */
  label: string
  /** Optional icon */
  icon?: ReactNode
}

export interface SortDropdownProps<T extends string> {
  /** Currently selected sort value */
  value: T
  /** Available sort options */
  options: SortOption<T>[]
  /** Callback when sort value changes */
  onChange: (value: T) => void
  /** Button variant (default: ghost) */
  variant?: 'ghost' | 'outline' | 'default'
  /** Button size (default: sm) */
  size?: 'sm' | 'default' | 'lg'
  /** Button className */
  className?: string
  /** Button label prefix (default: "Sort by") */
  label?: string
  /** Test ID for testing */
  testId?: string
}

/**
 * Dropdown for selecting sort order.
 *
 * Renders a button that opens a dropdown menu with sort options.
 * Highlights the currently active sort option.
 *
 * @example
 * ```tsx
 * <SortDropdown
 *   value={sortOrder}
 *   options={[
 *     { value: 'newest', label: 'Newest' },
 *     { value: 'oldest', label: 'Oldest' },
 *     { value: 'importance', label: 'Importance' },
 *   ]}
 *   onChange={setSortOrder}
 * />
 * ```
 */
export function SortDropdown<T extends string>({
  value,
  options,
  onChange,
  variant = 'ghost',
  size = 'sm',
  className,
  label = 'Sort by',
  testId,
}: SortDropdownProps<T>) {
  const currentOption = options.find(o => o.value === value)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant={variant}
          size={size}
          className={cn('gap-1.5', className)}
          data-testid={testId}
        >
          {label}: {currentOption?.label ?? value}
          <IconChevronDown className="h-3 w-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {options.map(option => (
          <DropdownMenuItem
            key={option.value}
            onClick={() => onChange(option.value)}
            className={cn(
              'flex items-center gap-2',
              option.value === value && 'bg-accent'
            )}
          >
            {option.icon}
            {option.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
