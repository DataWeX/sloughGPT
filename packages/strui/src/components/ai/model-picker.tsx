import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { ChevronDownIcon } from '../../lib/icons'
import { cn } from '../../lib/cn'

export interface ModelOption {
  id: string
  label: string
  /** Short tag (e.g. “fast”, “vision”). */
  badge?: string
  disabled?: boolean
}

export interface ModelPickerProps {
  value: string
  options: ModelOption[]
  onChange: (id: string) => void
  disabled?: boolean
  className?: string
  /** Trigger width — full width on mobile by default. */
  fullWidth?: boolean
}

/** Model / preset selector for agent and chat headers. */
export function ModelPicker({
  value,
  options,
  onChange,
  disabled,
  className,
  fullWidth = true,
}: ModelPickerProps) {
  const current = options.find((o) => o.id === value)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="outline"
          disabled={disabled}
          className={cn(
            'str-touch-target justify-between gap-2 font-normal',
            fullWidth && 'w-full min-w-0 sm:w-auto sm:min-w-[12rem]',
            className,
          )}
        >
          <span className="truncate">{current?.label ?? value}</span>
          <ChevronDownIcon className="shrink-0 opacity-70" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-[min(60dvh,24rem)] w-[var(--radix-dropdown-menu-trigger-width)] str-chat-scroll sm:min-w-[14rem]">
        {options.map((o) => (
          <DropdownMenuItem
            key={o.id}
            disabled={o.disabled}
            className="str-touch-target flex items-center justify-between gap-2"
            onSelect={() => onChange(o.id)}
          >
            <span className="truncate">{o.label}</span>
            {o.badge ? (
              <Badge variant="secondary" className="shrink-0 text-[10px] uppercase">
                {o.badge}
              </Badge>
            ) : null}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
