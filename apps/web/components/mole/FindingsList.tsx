'use client'

import { cn, FoldSection, StatusDot } from '@sloughgpt/strui'
import { SEVERITY_LABEL, SEVERITY_ORDER, SEVERITY_TEXT_CLASS, SEVERITY_TONE } from './severity'
import type { MoleFinding, MoleSeverity } from './types'

interface FindingsListProps {
  findings: MoleFinding[]
}

function findingLabel(severity: MoleSeverity, finding: MoleFinding): string {
  return `${SEVERITY_LABEL[severity]}: ${finding.check} — ${finding.message}`
}

/**
 * One finding: severity marker, component tag, check id, message — and the
 * supporting detail folded behind a disclosure so the page stays a summary
 * rather than a dump.
 */
function FindingRow({ finding, severity }: { finding: MoleFinding; severity: MoleSeverity }) {
  const heading = (
    <span className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
      <StatusDot tone={SEVERITY_TONE[severity]} aria-hidden />
      {finding.component ? (
        <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
          {finding.component}
        </span>
      ) : null}
      <code className="text-[10px] text-muted-foreground">{finding.check}</code>
      <span className="min-w-0 flex-1 text-sm text-foreground">{finding.message}</span>
    </span>
  )

  if (!finding.detail) {
    return (
      <li
        data-testid="finding-row"
        aria-label={findingLabel(severity, finding)}
        className="flex items-start gap-2 rounded-md border border-border/50 px-3 py-2"
      >
        {heading}
      </li>
    )
  }

  return (
    <li data-testid="finding-row">
      <FoldSection
        heading={heading}
        aria-label={`${findingLabel(severity, finding)} — detail`}
        data-testid="finding-detail"
        className="border-border/50 bg-transparent open:bg-muted/30"
      >
        <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">
          {finding.detail}
        </pre>
      </FoldSection>
    </li>
  )
}

/** Findings grouped worst-first: critical → warn → info → ok. */
export function FindingsList({ findings }: FindingsListProps) {
  const all = findings ?? []
  if (all.length === 0) {
    return (
      <p data-testid="findings-empty" className="py-6 text-center text-sm text-muted-foreground">
        No findings recorded.
      </p>
    )
  }

  const groups = SEVERITY_ORDER.map((severity) => ({
    severity,
    items: all.filter((f) => f.severity === severity),
  })).filter((group) => group.items.length > 0)

  return (
    <div data-testid="findings-list" className="space-y-4">
      {groups.map(({ severity, items }) => (
        <section
          key={severity}
          aria-label={`${SEVERITY_LABEL[severity]} findings`}
          className="space-y-1.5"
        >
          <h3
            className={cn(
              'flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider',
              SEVERITY_TEXT_CLASS[severity],
            )}
          >
            {SEVERITY_LABEL[severity]}
            <span className="text-muted-foreground">({items.length})</span>
          </h3>
          <ul className="space-y-1.5">
            {items.map((finding, index) => (
              <FindingRow
                key={`${finding.source}-${finding.check}-${index}`}
                finding={finding}
                severity={severity}
              />
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
