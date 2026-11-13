'use client'

/** Snapshots tab: save-named weight snapshots, load and delete them. */
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  TabsContent,
  IconPlus,
  IconRefresh,
  IconTrash,
} from '@sloughgpt/strui'
import { formatDateTimeFull } from '@/lib/time-format'
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function SnapshotsTab({ wb }: { wb: SoulsWorkbench }) {
  const {
    handleLoadSnapshot,
    handleLoadSnapshots,
    handleSaveSnapshot,
    newSnapshotName,
    setDeleteTarget,
    setDeleteType,
    setNewSnapshotName,
    snapshots,
  } = wb

  return (
    <TabsContent value="snapshots" className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Weight Snapshots</CardTitle>
          <Button
            size="sm"
            variant="ghost"
            onClick={handleLoadSnapshots}
            aria-label="Refresh snapshots"
          >
            <IconRefresh className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <Input
              value={newSnapshotName}
              onChange={(e) => setNewSnapshotName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSaveSnapshot()}
              placeholder="Snapshot name..."
            />
            <Button size="sm" onClick={handleSaveSnapshot} disabled={!newSnapshotName.trim()}>
              <IconPlus className="h-3.5 w-3.5 mr-1" /> Save
            </Button>
          </div>
          {snapshots.length === 0 ? (
            <div className="text-center py-6 space-y-1">
              <p className="text-sm text-muted-foreground">No snapshots saved yet.</p>
              <p className="text-xs text-muted-foreground">
                Save your current trait weights as a named snapshot to reuse later.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {snapshots.map((snap) => (
                <div
                  key={snap.name}
                  className="flex items-center justify-between rounded-md border border-border/40 px-3 py-2 text-sm group hover:bg-muted/50 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <span className="font-medium truncate">{snap.name}</span>
                    {snap.saved_at && (
                      <span className="text-xs text-muted-foreground ml-2">
                        {formatDateTimeFull(snap.saved_at)}
                      </span>
                    )}
                  </div>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                    <Button size="sm" variant="ghost" onClick={() => handleLoadSnapshot(snap.name)}>
                      Load
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive"
                      onClick={() => {
                        setDeleteTarget(snap.name)
                        setDeleteType('snapshot')
                      }}
                    >
                      <IconTrash className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </TabsContent>
  )
}
