'use client'

import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { cn } from '../../lib/cn'

/* ── Context ────────────────────────────────────────────────── */

interface TabsContextValue {
  value: string
  onValueChange: (value: string) => void
}

const TabsContext = createContext<TabsContextValue | null>(null)

function useTabsContext() {
  const ctx = useContext(TabsContext)
  if (!ctx) throw new Error('Tabs compound components must be used within <Tabs>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────── */

interface TabsRootProps {
  value?: string
  defaultValue?: string
  onValueChange?: (value: string) => void
  /** Alias for onValueChange */
  onChange?: (value: string) => void
  className?: string
  /** Shorthand for rendering tabs from a data array */
  tabs?: Array<{ value: string; label: string; count?: number; disabled?: boolean }>
  children?: ReactNode
}

function Tabs({
  value: controlledValue,
  defaultValue = '',
  onValueChange,
  onChange,
  className,
  tabs: tabDefs,
  children,
}: TabsRootProps) {
  const handler = onValueChange ?? onChange
  const [internalValue, setInternalValue] = useState(defaultValue)
  const isControlled = controlledValue !== undefined
  const value = isControlled ? controlledValue : internalValue

  const handleChange = useCallback(
    (v: string) => {
      if (!isControlled) setInternalValue(v)
      handler?.(v)
    },
    [isControlled, handler],
  )

  const ctx = { value, onValueChange: handleChange }

  if (tabDefs) {
    return (
      <TabsContext.Provider value={ctx}>
        <div className={className}>
          <TabsList>
            {tabDefs.map((t) => (
              <TabsTrigger key={t.value} value={t.value} disabled={t.disabled}>
                {t.label}
                {t.count !== undefined && (
                  <span className="ml-1.5 rounded-full bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                    {t.count}
                  </span>
                )}
              </TabsTrigger>
            ))}
          </TabsList>
          {children}
        </div>
      </TabsContext.Provider>
    )
  }

  return (
    <TabsContext.Provider value={ctx}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  )
}

/* ── List ───────────────────────────────────────────────────── */

const TabsList = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      role="tablist"
      className={cn(
        'inline-flex h-10 items-center justify-start gap-0.5 rounded-lg border border-border bg-muted/50 p-1 text-muted-foreground',
        className,
      )}
      {...props}
    />
  ),
)
TabsList.displayName = 'TabsList'

/* ── Trigger ────────────────────────────────────────────────── */

interface TabsTriggerProps extends HTMLAttributes<HTMLButtonElement> {
  value: string
  disabled?: boolean
}

const TabsTrigger = forwardRef<HTMLButtonElement, TabsTriggerProps>(
  ({ className, value: triggerValue, disabled, ...props }, ref) => {
    const { value, onValueChange } = useTabsContext()
    const isActive = value === triggerValue

    return (
      <button
        ref={ref}
        type="button"
        role="tab"
        aria-selected={isActive}
        aria-disabled={disabled}
        data-state={isActive ? 'active' : 'inactive'}
        disabled={disabled}
        onClick={() => !disabled && onValueChange(triggerValue)}
        className={cn(
          'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-all duration-150',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2',
          'disabled:pointer-events-none disabled:opacity-40',
          isActive
            ? 'bg-background text-foreground shadow-sm'
            : 'text-muted-foreground hover:text-foreground hover:bg-background/50',
          className,
        )}
        {...props}
      />
    )
  },
)
TabsTrigger.displayName = 'TabsTrigger'

/* ── Content ────────────────────────────────────────────────── */

interface TabsContentProps extends HTMLAttributes<HTMLDivElement> {
  value: string
  /** Keep the content in DOM (hidden) when inactive — useful for forms */
  forceMount?: boolean
}

const TabsContent = forwardRef<HTMLDivElement, TabsContentProps>(
  ({ className, value: contentValue, forceMount = false, ...props }, ref) => {
    const { value } = useTabsContext()
    const isActive = value === contentValue

    if (!forceMount && !isActive) return null

    return (
      <div
        ref={ref}
        role="tabpanel"
        tabIndex={0}
        hidden={forceMount ? !isActive : undefined}
        className={cn(
          'mt-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2',
          className,
        )}
        {...props}
      />
    )
  },
)
TabsContent.displayName = 'TabsContent'

export { Tabs, TabsList, TabsTrigger, TabsContent }
