/* Planner client — browser-side fetch wrappers for /api/planner/* routes. */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlannerCard {
  id: string;
  title: string;
  description: string;
  column: string;
  priority: string;
  tags: string[];
  due_date: string;
  created_at: string;
  updated_at: string;
}

export interface PlannerNote {
  id: string;
  title: string;
  body: string;
  status: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface Board {
  columns: Record<string, PlannerCard[]>;
  wip_limits: Record<string, number>;
}

export interface TagCount {
  tag: string;
  count: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/planner${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Planner API error: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function fetchBoard(): Promise<{ board: Board }> {
  return api("/board");
}

export async function moveCard(payload: {
  card_id: string;
  column: string;
}): Promise<void> {
  await api("/board/move", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function syncNotes(): Promise<void> {
  await api("/sync", { method: "POST" });
}

export async function createNote(payload: {
  title: string;
  tags?: string[];
  status?: string;
  body?: string;
}): Promise<{ note: PlannerNote }> {
  return api("/notes", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function updateNote(
  id: string,
  payload: {
    title?: string;
    body?: string;
    status?: string;
    tags?: string[];
  },
): Promise<{ note: PlannerNote }> {
  return api(`/notes/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function deleteNote(id: string): Promise<void> {
  await api(`/notes/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function fetchTags(): Promise<{ tags: TagCount[] }> {
  return api("/tags");
}
