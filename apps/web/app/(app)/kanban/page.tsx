/* Kanban page — server component. Fetches from planner backend, renders
   pre-built Kanban components. Swap components later without changing fetch logic. */

import { KanbanBoard } from "@/components/kanban/KanbanBoard";
import type { KanbanBoard as KanbanBoardData } from "@/components/kanban/types";

const PLANNER_URL = process.env.PLANNER_URL || "http://127.0.0.1:8787";

async function fetchBoard(): Promise<KanbanBoardData> {
  const res = await fetch(`${PLANNER_URL}/api/board`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Planner backend returned ${res.status}`);
  }
  const data = await res.json();
  return data.board;
}

export default async function KanbanPage() {
  let board: KanbanBoardData;
  let error: string | null = null;

  try {
    board = await fetchBoard();
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load board";
    board = { name: "board", columns: [], cards: [] };
  }

  return (
    <div className="mx-auto max-w-7xl p-4">
      <div className="mb-4 flex items-center justify-between">
        <h1
          className="text-base font-medium"
          style={{ color: "rgb(var(--color-foreground))" }}
        >
          Kanban Board
        </h1>
        {error && (
          <p
            className="text-sm"
            style={{ color: "rgb(var(--color-error))" }}
          >
            {error}
          </p>
        )}
      </div>

      {board.columns.length === 0 ? (
        <div
          className="rounded-lg border border-dashed p-8 text-center"
          style={{
            borderColor: "rgb(var(--color-border))",
            color: "rgb(var(--color-muted))",
          }}
        >
          <p className="text-sm">
            No board found. Start the planner backend:
          </p>
          <code
            className="mt-2 block text-xs"
            style={{ color: "rgb(var(--color-foreground))" }}
          >
            planner gui --port 8787
          </code>
        </div>
      ) : (
        <KanbanBoard board={board} />
      )}
    </div>
  );
}
