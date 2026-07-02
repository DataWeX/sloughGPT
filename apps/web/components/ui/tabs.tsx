'use client'

import { createContext, forwardRef, useCallback, useContext, type HTMLAttributes, type ReactNode } from 'react'
import { cn } from '@/lib/cn'

/* ── Context ────────────────────────────────────────────────────── */

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

/* ── Root ───────────────────────────────────────────────────────── */

interface TabsRootProps {
  value?: string
  defaultValue?: string
  onValueChange?: (value: string) => void
  onChange?: (value: string) => void
  className?: string
  tabs?: Array<{ value: string; label: string; count?: number }>
  children?: ReactNode
}

function Tabs({ value: controlledValue, defaultValue = '', onValueChange, onChange, className, tabs: tabDefs, children }: TabsRootProps) {
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

  if (tabDefs) {
    return (
      <TabsContext.Provider value={{ value, onValueChange: handleChange }}>
        <div className={className}>
          <TabsList>
            {tabDefs.map(t => (
              <TabsTrigger key={t.value} value={t.value}>
                {t.label}{t.count !== undefined ? ` (${t.count})` : ''}
              </TabsTrigger>
            ))}
          </TabsList>
          {children}
        </div>
      </TabsContext.Provider>
    )
  }

  return (
    <TabsContext.Provider value={{ value, onValueChange: handleChange }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  )
}

import { useState } from 'react'

/* ── List ───────────────────────────────────────────────────────── */

const TabsList = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      role="tablist"
      className={cn(
        'inline-flex h-10 items-center justify-center gap-1 rounded-lg border border-border bg-muted/50 p-1 text-muted-foreground',
        className,
      )}
      {...props}
    />
  ),
)
TabsList.displayName = 'TabsList'

/* ── Trigger ────────────────────────────────────────────────────── */

interface TabsTriggerProps extends HTMLAttributes<HTMLButtonElement> {
  value: string
}

const TabsTrigger = forwardRef<HTMLButtonElement, TabsTriggerProps>(
  ({ className, value: triggerValue, ...props }, ref) => {
    const { value, onValueChange } = useTabsContext()
    const isActive = value === triggerValue

    return (
      <button
        ref={ref}
        type="button"
        role="tab"
        aria-selected={isActive}
        data-state={isActive ? 'active' : 'inactive'}
        onClick={() => onValueChange(triggerValue)}
        className={cn(
          'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-all duration-150',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2',
          'disabled:pointer-events-none disabled:opacity-50',
          isActive ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
          className,
        )}
        {...props}
      />
    )
  },
)
TabsTrigger.displayName = 'TabsTrigger'

/* ── Content ────────────────────────────────────────────────────── */

interface TabsContentProps extends HTMLAttributes<HTMLDivElement> {
  value: string
}

const TabsContent = forwardRef<HTMLDivElement, TabsContentProps>(
  ({ className, value: contentValue, ...props }, ref) => {
    const { value } = useTabsContext()
    if (value !== contentValue) return null

    return (
      <div
        ref={ref}
        role="tabpanel"
        tabIndex={0}
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
