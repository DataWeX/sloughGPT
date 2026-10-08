/**
 * Shared metadata for search result stores — the single source used by
 * the workspace search page (grouped cards) and the command palette
 * (flat results). Keys MUST match the API's hit.store values.
 */

export interface SearchStoreMeta {
  label: string
  color: string
  link: string
  icon: string
}

export const SEARCH_STORES: Record<string, SearchStoreMeta> = {
  members: {
    label: 'Members',
    color: 'bg-primary/15 text-primary dark:bg-primary/10 dark:text-primary',
    link: '/workspace/members',
    icon: '👥',
  },
  training_jobs: {
    label: 'Training Jobs',
    color: 'bg-info/15 text-info dark:bg-info/10 dark:text-info',
    link: '/training',
    icon: '🧠',
  },
  datasets: {
    label: 'Datasets',
    color: 'bg-success/15 text-success dark:bg-success/10 dark:text-success',
    link: '/datasets',
    icon: '🗄️',
  },
  knowledge: {
    label: 'Knowledge',
    color: 'bg-accent text-accent dark:bg-accent/30 dark:text-accent',
    link: '/knowledge',
    icon: '📖',
  },
  files: {
    label: 'Files',
    color: 'bg-info/10 text-info dark:bg-info/15 dark:text-info',
    link: '/files',
    icon: '📁',
  },
  kb: {
    label: 'Knowledge Base',
    color: 'bg-primary/10 text-primary dark:bg-primary/15 dark:text-primary',
    link: '/kb',
    icon: '🗃️',
  },
  memory: {
    label: 'Memory',
    color: 'bg-warning/10 text-warning dark:bg-warning/15 dark:text-warning',
    link: '/memory',
    icon: '💭',
  },
  sessions: {
    label: 'Chat Sessions',
    color: 'bg-primary/15 text-primary dark:bg-primary/10 dark:text-primary',
    link: '/chat',
    icon: '💬',
  },
  docs: {
    label: 'Docs',
    color: 'bg-warning/15 text-warning dark:bg-warning/10 dark:text-warning',
    link: '', // file-backed: no viewer — rendered as a non-navigable group
    icon: '📄',
  },
  dev_notes: {
    label: 'Dev Notes',
    color: 'bg-info/15 text-info dark:bg-info/10 dark:text-info',
    link: '',
    icon: '📝',
  },
  kanban_cards: {
    label: 'Kanban',
    color: 'bg-success/15 text-success dark:bg-success/10 dark:text-success',
    link: '',
    icon: '🗂️',
  },
}
