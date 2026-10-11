/**
 * Primary dashboard actions — Seed / Reflect / Export / Import.
 *
 * Rendered into `PageContainer`'s `headerRight` slot. Each button carries its
 * own busy flag so a single in-flight action disables only itself.
 */

import { Button } from '@sloughgpt/strui'

export interface DashboardActionsProps {
  onSeed: () => void
  seeding: boolean
  onReflect: () => void
  reflecting: boolean
  onExport: () => void
  exporting: boolean
  onImport: () => void
  importing: boolean
}

export function DashboardActions({
  onSeed,
  seeding,
  onReflect,
  reflecting,
  onExport,
  exporting,
  onImport,
  importing,
}: DashboardActionsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      <Button size="sm" variant="outline" onClick={onSeed} disabled={seeding}>
        {seeding ? 'Seeding...' : 'Seed 30'}
      </Button>
      <Button size="sm" variant="outline" onClick={onReflect} disabled={reflecting}>
        {reflecting ? 'Reflecting...' : 'Reflect'}
      </Button>
      <Button size="sm" variant="outline" onClick={onExport} disabled={exporting}>
        {exporting ? 'Exporting...' : 'Export'}
      </Button>
      <Button size="sm" variant="outline" onClick={onImport} disabled={importing}>
        {importing ? 'Importing...' : 'Import'}
      </Button>
    </div>
  )
}
