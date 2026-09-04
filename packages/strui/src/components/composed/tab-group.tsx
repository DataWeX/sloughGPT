'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/tabs'

export interface TabItem {
  /** Tab value (unique identifier) */
  value: string
  /** Tab label */
  label: string
  /** Tab content */
  content: ReactNode
  /** Whether the tab is disabled */
  disabled?: boolean
}

export interface TabGroupProps {
  /** Array of tab configurations */
  tabs: TabItem[]
  /** Default active tab value */
  defaultValue?: string
  /** Controlled active tab value */
  value?: string
  /** Callback when tab changes */
  onValueChange?: (value: string) => void
  /** Tab list layout */
  layout?: 'underline' | 'pills' | 'boxed'
  /** Additional CSS classes for the container */
  className?: string
  /** Additional CSS classes for the tab list */
  listClassName?: string
  /** Test ID for testing */
  testId?: string
}

/**
 * Consistent tab group with content panels.
 *
 * Renders tabs with a consistent layout and handles tab switching.
 * Supports underline, pills, and boxed tab styles.
 *
 * @example
 * ```tsx
 * <TabGroup
 *   tabs={[
 *     { value: 'overview', label: 'Overview', content: <OverviewPanel /> },
 *     { value: 'details', label: 'Details', content: <DetailsPanel /> },
 *     { value: 'logs', label: 'Logs', content: <LogsPanel /> },
 *   ]}
 *   defaultValue="overview"
 * />
 * ```
 */
export function TabGroup({
  tabs,
  defaultValue,
  value,
  onValueChange,
  layout = 'underline',
  className,
  listClassName,
  testId,
}: TabGroupProps) {
  const listStyles = {
    underline: 'border-b border-border',
    pills: 'bg-muted p-1 rounded-lg',
    boxed: 'border border-border rounded-lg p-1',
  }

  const triggerStyles = {
    underline: 'data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none',
    pills: 'data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-md',
    boxed: 'data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-md',
  }

  return (
    <Tabs
      defaultValue={defaultValue}
      value={value}
      onValueChange={onValueChange}
      className={className}
      data-testid={testId}
    >
      <TabsList className={cn(listStyles[layout], listClassName)}>
        {tabs.map(tab => (
          <TabsTrigger
            key={tab.value}
            value={tab.value}
            disabled={tab.disabled}
            className={triggerStyles[layout]}
          >
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {tabs.map(tab => (
        <TabsContent key={tab.value} value={tab.value}>
          {tab.content}
        </TabsContent>
      ))}
    </Tabs>
  )
}
