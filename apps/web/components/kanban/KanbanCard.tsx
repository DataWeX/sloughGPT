/* KanbanCard — server-rendered card. Replace with custom component later. */

import type { KanbanCardData } from "./types";

const PRIORITY_COLORS: Record<string, string> = {
  low: "rgb(var(--color-muted))",
  medium: "rgb(var(--color-accent))",
  high: "rgb(var(--color-warning))",
  critical: "rgb(var(--color-error))",
};

export function KanbanCard({ card }: { card: KanbanCardData }) {
  return (
    <div
      className="rounded-lg border p-3"
      style={{
        borderColor: "rgb(var(--color-border))",
        backgroundColor: "rgb(var(--color-surface))",
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <h4
          className="text-sm font-medium leading-tight"
          style={{ color: "rgb(var(--color-foreground))" }}
        >
          {card.title}
        </h4>
        <span
          className="h-2 w-2 shrink-0 rounded-full"
          style={{
            backgroundColor: PRIORITY_COLORS[card.priority] ?? "rgb(var(--color-muted))",
          }}
        />
      </div>

      {card.description && (
        <p
          className="mt-1 text-xs leading-relaxed"
          style={{ color: "rgb(var(--color-muted))" }}
        >
          {card.description.length > 120
            ? `${card.description.slice(0, 120)}...`
            : card.description}
        </p>
      )}

      {card.tags.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {card.tags.map((tag) => (
            <span
              key={tag}
              className="rounded px-1.5 py-0.5 text-[10px] font-medium"
              style={{
                backgroundColor: "rgb(var(--color-accent) / 0.15)",
                color: "rgb(var(--color-accent))",
              }}
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {(card.assignee || card.due_date) && (
        <div
          className="mt-2 flex items-center gap-2 text-[10px]"
          style={{ color: "rgb(var(--color-muted))" }}
        >
          {card.assignee && <span>{card.assignee}</span>}
          {card.due_date && <span>{card.due_date}</span>}
        </div>
      )}
    </div>
  );
}
