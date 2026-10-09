import { cn } from '../../lib/cn'
import { FEATURE_VERSIONS, type FeatureName } from '../../versions'

export interface FeatureVersionEntry {
  name: string
  backend: string
  struiVersion: string
  backendVersion: string | null
  componentCount: number
}

export interface VersionInspectorProps {
  /** Backend feature versions from GET /health/detailed → versions.features */
  backendFeatures?: Record<string, { backend: string; api: string }>
  className?: string
}

function compareVersions(a: string, b: string): 'match' | 'mismatch' | 'unknown' {
  if (!a || !b || a === '0.0.0' || b === '0.0.0') return 'unknown'
  return a === b ? 'match' : 'mismatch'
}

function VersionDot({ status }: { status: 'match' | 'mismatch' | 'unknown' }) {
  const color = status === 'match' ? 'bg-success' : status === 'mismatch' ? 'bg-warning' : 'bg-muted-foreground/50'
  return <span className={cn('inline-block w-2 h-2 rounded-full shrink-0', color)} />
}

function VersionRow({ entry }: { entry: FeatureVersionEntry }) {
  const status = compareVersions(entry.struiVersion, entry.backendVersion ?? '')
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <div className="flex items-center gap-2 min-w-0">
        <VersionDot status={status} />
        <span className="font-medium truncate">{entry.name}</span>
        <span className="text-muted-foreground text-xs truncate">{entry.backend}</span>
      </div>
      <div className="flex items-center gap-3 shrink-0 text-xs font-mono">
        <span className="text-muted-foreground">ui:{entry.struiVersion}</span>
        <span className="text-muted-foreground">be:{entry.backendVersion ?? '—'}</span>
        <span className="text-muted-foreground">{entry.componentCount}c</span>
      </div>
    </div>
  )
}

/**
 * VersionInspector — displays the alignment between strui component versions
 * and backend feature versions.
 */
export function VersionInspector({ backendFeatures, className }: VersionInspectorProps) {
  const entries: FeatureVersionEntry[] = Object.entries(FEATURE_VERSIONS).map(([name, info]) => ({
    name,
    backend: info.backend,
    struiVersion: info.api,
    backendVersion: backendFeatures?.[name]?.api ?? null,
    componentCount: info.components.length,
  }))

  return (
    <div className={cn('divide-y divide-border/50', className)}>
      {entries.map(entry => (
        <VersionRow key={entry.name} entry={entry} />
      ))}
    </div>
  )
}

export default VersionInspector
