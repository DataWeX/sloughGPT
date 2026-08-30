/* KanbanColumn — server-rendered column. Replace with custom component later. */

import type { KanbanColumn as KanbanColumnData, KanbanCardData } from "./types";
import { KanbanCard } from "./KanbanCard";

export function KanbanColumn({
  column,
  cards,
}: {
  column: KanbanColumnData;
  cards: KanbanCardData[];
}) {
  const overWip = column.wip_limit > 0 && cards.length > column.wip_limit;

  return (
    <div className="flex min-w-[280px] flex-col">
      {/* Column header */}
      <div
        className="mb-3 flex items-center justify-between rounded-lg px-3 py-2"
        style={{ backgroundColor: "rgb(var(--color-surface))" }}
      >
        <div className="flex items-center gap-2">
          <h3
            className="text-sm font-medium capitalize"
            style={{ color: "rgb(var(--color-foreground))" }}
          >
            {column.name.replace(/_/g, " ")}
          </h3>
          <span
            className="rounded-full px-1.5 py-0.5 text-[10px] font-medium"
            style={{
              backgroundColor: overWip
                ? "rgb(var(--color-error) / 0.15)"
                : "rgb(var(--color-muted) / 0.15)",
              color: overWip
                ? "rgb(var(--color-error))"
                : "rgb(var(--color-muted))",
            }}
          >
            {cards.length}
            {column.wip_limit > 0 ? ` / ${column.wip_limit}` : ""}
          </span>
        </div>
      </div>

      {/* Cards */}
      <div className="flex flex-col gap-2">
        {cards.length === 0 ? (
          <div
            className="rounded-lg border border-dashed p-4 text-center text-xs"
            style={{
              borderColor: "rgb(var(--color-border))",
              color: "rgb(var(--color-muted))",
            }}
          >
            No cards
          </div>
        ) : (
          cards.map((card) => <KanbanCard key={card.id} card={card} />)
        )}
      </div>
    </div>
  );
}
