'use client'

/** Page header row: title, compare toggle, save and reset actions. */
import { Button } from '@sloughgpt/strui'
import type { PersonalityWorkbench } from './usePersonalityWorkbench'

export function PersonalityHeader({ wb }: { wb: PersonalityWorkbench }) {
  const { handleExport, handleImport, handleReset, handleSave, saving } = wb

  return (
    <>
      {/* Header with save/reset */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">Personalities</h2>
          <p className="text-xs sm:text-sm text-muted-foreground">
            Switch personalities and define how your agent speaks and behaves
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={handleExport}>
            Export
          </Button>
          <Button variant="outline" onClick={handleImport}>
            Import
          </Button>
          <Button variant="outline" onClick={handleReset}>
            Reset
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>
    </>
  )
}
