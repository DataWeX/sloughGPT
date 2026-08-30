/* KanbanBoard — server-rendered board. Replace with custom component later. */

import type { KanbanBoard as KanbanBoardData } from "./types";
import { KanbanColumn } from "./KanbanColumn";

export function KanbanBoard({ board }: { board: KanbanBoardData }) {
  const cardsByColumn = new Map<string, typeof board.cards>();

  for (const col of board.columns) {
    cardsByColumn.set(col.name, []);
  }
  for (const card of board.cards) {
    const list = cardsByColumn.get(card.column);
    if (list) {
      list.push(card);
    } else {
      cardsByColumn.set(card.column, [card]);
    }
  }

  return (
    <div
      className="flex gap-4 overflow-x-auto pb-4"
      style={{ minHeight: "calc(100vh - 200px)" }}
    >
      {board.columns
        .sort((a, b) => a.order - b.order)
        .map((col) => (
          <KanbanColumn
            key={col.name}
            column={col}
            cards={cardsByColumn.get(col.name) ?? []}
          />
        ))}
    </div>
  );
}
