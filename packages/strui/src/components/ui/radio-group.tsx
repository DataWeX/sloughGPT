'use client'

import * as React from 'react'
import { cn } from '../../lib/cn'

export interface RadioGroupProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'onChange'> {
  value?: string
  onValueChange?: (value: string) => void
  name?: string
  orientation?: 'vertical' | 'horizontal'
}

const RadioGroup = React.forwardRef<HTMLDivElement, RadioGroupProps>(
  (
    {
      className,
      value,
      onValueChange,
      name,
      orientation = 'vertical',
      children,
      ...props
    },
    ref,
  ) => {
    const groupName = React.useId()

    return (
      <div
        ref={ref}
        role="radiogroup"
        aria-orientation={orientation}
        className={cn(
          orientation === 'vertical' ? 'flex flex-col gap-2' : 'flex flex-row flex-wrap gap-2',
          className,
        )}
        {...props}
      >
        {React.Children.map(children, (child) => {
          if (!React.isValidElement(child)) return child
          const childValue = (child.props as { value?: string }).value
          return (
            <child.type
              {...child.props}
              name={name ?? groupName}
              checked={value !== undefined ? childValue === value : undefined}
              onCheckedChange={onValueChange
                ? () => onValueChange(childValue as string)
                : undefined}
            />
          )
        })}
      </div>
    )
  },
)
RadioGroup.displayName = 'RadioGroup'

export { RadioGroup }
