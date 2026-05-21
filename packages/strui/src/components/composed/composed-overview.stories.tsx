import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Switch } from '../ui/switch'
import { AppShell } from './app-shell'
import { EmptyCard } from './empty-card'
import { FormField } from './form-field'
import { InlineBanner } from './inline-banner'
import { KpiGrid } from './kpi-grid'
import { NavRail, NavRailLink } from './nav-rail'
import { PageHeader } from './page-header'
import { SearchInput } from './search-input'
import { SettingsRow } from './settings-row'
import { Skeleton } from './skeleton'
import { StatCard } from './stat-card'
import { Toolbar } from './toolbar'

const meta = {
  title: 'Composed/Overview',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        story:
          'Application shell patterns: nav rail, page header, KPI grid, forms, banners, and settings rows built from strui primitives.',
      },
    },
  },
} satisfies Meta

export default meta

export const DashboardShell: StoryObj = {
  render: () => {
    const [on, setOn] = useState(true)
    return (
      <AppShell
        sidebar={
          <NavRail header={<span className="px-3 text-xs font-bold uppercase tracking-wider">Acme</span>}>
            <NavRailLink href="#" active>
              Home
            </NavRailLink>
            <NavRailLink href="#">Agents</NavRailLink>
            <NavRailLink href="#">Settings</NavRailLink>
          </NavRail>
        }
      >
        <PageHeader
          title="Overview"
          description="Composed building blocks for dashboards and AI apps."
          actions={
            <>
              <Button size="sm">New</Button>
              <Button size="sm" variant="secondary">
                Export
              </Button>
            </>
          }
        />
        <Toolbar>
          <SearchInput className="max-w-sm" placeholder="Filter…" />
          <Button size="sm" variant="ghost">
            Refresh
          </Button>
        </Toolbar>
        <div className="str-safe-x space-y-6 p-4">
          <InlineBanner
            variant="warning"
            title="Rate limit"
            description="You are approaching the tier cap for this workspace."
            action={<Button size="sm">Upgrade</Button>}
          />
          <KpiGrid>
            <StatCard label="Requests" value="12.4k" icon="📈" trend={{ value: 3, positive: true }} />
            <StatCard label="Latency p95" value="420ms" />
            <StatCard label="Errors" value="0.02%" />
            <StatCard label="Cost" value="$48" />
          </KpiGrid>
          <div className="grid gap-6 lg:grid-cols-2">
            <EmptyCard message="No experiments" action={<Button className="str-touch-target w-full sm:w-auto">Create experiment</Button>} />
            <div className="space-y-3">
              <p className="text-xs font-medium uppercase text-muted-foreground">Loading skeleton</p>
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-[75%]" />
              <Skeleton className="h-24 w-full" />
            </div>
          </div>
          <SettingsRow
            title="Enable streaming"
            description="Show tokens as they are generated."
            control={<Switch checked={on} onCheckedChange={setOn} />}
          />
          <FormField id="api" label="API key" hint="Stored in the browser for this demo.">
            <Input id="api" placeholder="sk-…" autoComplete="off" />
          </FormField>
        </div>
      </AppShell>
    )
  },
}
