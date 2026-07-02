import { forwardRef } from 'react'

import { SearchIcon } from '../../lib/icons'
import { Input, type InputProps } from '../ui/input'
import { cn } from '../../lib/cn'

export interface SearchInputProps extends Omit<InputProps, 'type'> {
  /** Visually hide the search icon (still keep padding for alignment). */
  hideIcon?: boolean
}

/** Search field with leading icon — suitable for command palettes and dataset pickers. */
export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  ({ className, hideIcon, placeholder = 'Search…', ...props }, ref) => {
    return (
      <div className="relative w-full">
        {!hideIcon ? (
          <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" aria-hidden="true">
            <SearchIcon />
          </span>
        ) : null}
        <Input
          ref={ref}
          type="search"
          placeholder={placeholder}
          autoComplete="off"
          className={cn(!hideIcon && 'pl-10', className)}
          aria-label={placeholder}
          {...props}
        />
      </div>
    )
  },
)
SearchInput.displayName = 'SearchInput'
