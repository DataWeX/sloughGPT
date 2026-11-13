'use client'

/**
 * Model lineage — which datasets/models produced which artifacts.
 *
 * Composes onto the Models tab (no new page): reads the registry's
 * artifact index with ?lineage=true and renders each stamped artifact as
 * a parent-chip -> child row. Roots without stamps are counted, never
 * listed — a family tree only shows links that exist.
 */
import { notify } from '@/lib/notify'
import { useState, useEffect } from 'react'
import { Badge, Card, CardContent, CardHeader, CardTitle, Skeleton } from '@sloughgpt/strui'
import { registryController, type ArtifactRecord } from '@/lib/registry-controller'

export default function ModelLineageCard() {
  const [artifacts, setArtifacts] = useState<ArtifactRecord[]>([])
  const [loading, setLoading] = useState(true)
  const addToast = notify.toast

  useEffect(() => {
    let active = true
    const run = async () => {
      try {
        const result = await registryController.artifacts()
        if (active) setArtifacts(result)
      } catch {
        if (active) {
          addToast('Could not load model lineage', 'error')
          setArtifacts([])
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void run()
    return () => {
      active = false
    }
  }, [addToast])

  const stamped = artifacts.filter((a) => (a.parents?.length ?? 0) > 0)

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base font-medium">Lineage</CardTitle>
        {artifacts.length > 0 && (
          <span className="text-xs text-muted-foreground font-mono">
            {stamped.length} of {artifacts.length} stamped
          </span>
        )}
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2" aria-busy="true" aria-label="Loading lineage">
            <Skeleton className="h-5 w-2/3" />
            <Skeleton className="h-5 w-1/2" />
          </div>
        ) : stamped.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No lineage stamped yet. New training exports and checkpoints record their parent
            datasets automatically ({'<file>.prov.json'} sidecars); pre-existing artifacts are roots
            until re-exported.
          </p>
        ) : (
          <ul className="space-y-2">
            {stamped.map((a) => (
              <li key={a.artifact_id} className="flex flex-wrap items-center gap-1.5 text-sm">
                {a.parents!.map((p) => (
                  <Badge
                    key={p}
                    variant="secondary"
                    className="font-mono text-xs"
                    data-testid={`lineage-parent-${p}`}
                  >
                    {p}
                  </Badge>
                ))}
                <span aria-hidden className="text-muted-foreground">
                  &rarr;
                </span>
                <span className="font-mono text-sm">{a.name}</span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}
