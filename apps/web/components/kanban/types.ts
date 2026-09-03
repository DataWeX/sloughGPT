/* Kanban types — shared between server and client components. */

export interface KanbanColumn {
  name: string;
  wip_limit: number;
  order: number;
}

export interface KanbanCardData {
  id: string;
  title: string;
  description: string;
  column: string;
  priority: string;
  tags: string[];
  due_date: string;
  assignee: string;
  notes: { id: string; text: string; author: string }[];
}

export interface KanbanBoard {
  name: string;
  columns: KanbanColumn[];
  cards: KanbanCardData[];
}

export interface TagCount {
  name: string;
  count: number;
}
