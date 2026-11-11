'use client'

/**
 * Dependencies tab — the engineering library landscape (card 20261111_110106).
 *
 * Reads `@/lib/deps.gen`, a BUILD-TIME artifact generated from both trees'
 * manifests (scripts/gen_deps_artifact.py). Same process as the build, so no
 * HTTP endpoint is earned — the boundary rule's "same process → function
 * call" half. Nothing here fetches.
 *
 * Layout follows the summarize-don't-dump rule: counts first, drift (the
 * answer to "where do the trees disagree") made visually obvious, raw
 * inventories one click away in collapsed folds.
 */
import {
  Card,
  CardContent,
  FoldSection,
  KpiGrid,
  StatCard,
  StatusBadge,
} from '@sloughgpt/strui'
import { DEPS, type DepGroup } from '@/lib/deps.gen'

const HEADING = 'text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block'

function uniqueCount(tree: 'python' | 'js'): number {
  return new Set(DEPS.entries.filter((e) => e.tree === tree).map((e) => e.key)).size
}

function GroupRow({ group }: { group: DepGroup }) {
  const manifests = [...new Set(group.entries.map((e) => e.manifest))]
  return (
    <li className="flex flex-wrap items-baseline justify-between gap-2 py-1">
      <span className="font-mono text-xs text-foreground">{group.key}</span>
      <span className="flex flex-wrap items-center gap-1.5">
        {group.versions.map((version) => (
          <code key={version} className="rounded bg-muted px-1 py-0.5 text-[10px] text-foreground">
            {version || 'unpinned'}
          </code>
        ))}
        <span className="text-[10px] text-muted-foreground">{manifests.length} manifests</span>
      </span>
    </li>
  )
}

function Inventory({ tree, label }: { tree: 'python' | 'js'; label: string }) {
  const entries = DEPS.entries.filter((e) => e.tree === tree)
  const byManifest = new Map<string, typeof entries>()
  for (const entry of entries) {
    const list = byManifest.get(entry.manifest) ?? []
    list.push(entry)
    byManifest.set(entry.manifest, list)
  }
  return (
    <FoldSection heading={`${label} inventory (${entries.length})`} open={false}>
      <div className="space-y-3">
        {[...byManifest.entries()].map(([manifest, list]) => (
          <div key={manifest}>
            <div className="text-xs font-medium text-foreground">{manifest}</div>
            <ul className="mt-1 grid grid-cols-1 gap-x-4 sm:grid-cols-2">
              {list.map((entry) => (
                <li key={`${entry.name}:${entry.kind}`} className="flex justify-between gap-2 text-xs">
                  <span className="font-mono text-muted-foreground">{entry.name}</span>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {entry.version || '*'}
                    {entry.kind !== 'runtime' && <span className="ml-1 opacity-70">{entry.kind}</span>}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </FoldSection>
  )
}

export function DevDepsTab() {
  const drift = DEPS.drift
  const crossTree = DEPS.cross_tree

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Generated at build from {DEPS.manifests.length} manifests — inventory only, no
        vulnerability feed, nothing fetched at runtime.
      </p>

      <KpiGrid columns={4}>
        <StatCard label="Python packages" value={uniqueCount('python')} numeric />
        <StatCard label="JavaScript packages" value={uniqueCount('js')} numeric />
        <StatCard label="Manifests" value={DEPS.manifests.length} numeric />
        <StatCard label="Drift" value={drift.length} numeric />
      </KpiGrid>

      {drift.length > 0 && (
        <Card className="p-3">
          <span className={HEADING}>Version drift</span>
          <CardContent className="p-0">
            <p className="mb-1 text-[10px] text-muted-foreground">
              Same package declared with different specs — the trees disagree:
            </p>
            <ul className="divide-y divide-border/50">
              {drift.map((group) => (
                <GroupRow key={group.key} group={group} />
              ))}
            </ul>
            <div className="mt-2">
              <StatusBadge tone="warning">drift</StatusBadge>
            </div>
          </CardContent>
        </Card>
      )}

      {crossTree.length > 0 && (
        <Card className="p-3">
          <span className={HEADING}>Both trees</span>
          <CardContent className="p-0">
            <p className="mb-1 text-[10px] text-muted-foreground">
              Declared by python AND js manifests:
            </p>
            <ul className="flex flex-wrap gap-1.5">
              {crossTree.map((key) => (
                <li key={key}>
                  <StatusBadge tone="info">{key}</StatusBadge>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <FoldSection heading={`Duplicates (${DEPS.duplicates.length})`} open={false}>
        <ul className="divide-y divide-border/50">
          {DEPS.duplicates.map((group) => (
            <GroupRow key={group.key} group={group} />
          ))}
        </ul>
      </FoldSection>

      <Inventory tree="python" label="Python" />
      <Inventory tree="js" label="JavaScript" />
    </div>
  )
}
