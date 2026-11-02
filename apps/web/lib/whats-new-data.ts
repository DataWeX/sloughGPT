export interface WhatsNewItem {
  id: string
  title: string
  description: string
  icon: string
  date: string
  href?: string
  tags?: string[]
}

export const whatsNewItems: WhatsNewItem[] = [
  {
    id: 'workspace-switcher',
    title: 'New workspace switcher',
    description:
      'Switching workspaces is now a lightweight inline switcher — faster to open and no dedicated pages to load.',
    icon: '📁',
    date: '2026-10-30',
    tags: ['Navigation', 'UX'],
  },
  {
    id: 'kv-cache-session',
    title: 'Faster multi-turn API calls',
    description:
      'The /v1 chat/completions endpoint now accepts a session_id, so API clients reuse the KV-cache across turns instead of recomputing every request.',
    icon: '⚡',
    date: '2026-10-29',
    tags: ['API', 'Performance'],
  },
  {
    id: 'fail-closed-auth',
    title: 'Security hardening',
    description:
      'Authentication now fails closed — if auth is misconfigured, requests are denied rather than allowed through. Safer defaults across the app.',
    icon: '🔒',
    date: '2026-10-28',
    tags: ['Security'],
  },
  {
    id: 'rag-chunking',
    title: 'Better knowledge recall',
    description:
      'Knowledge injection now chunks documents with fine-grained character windows sized to the prompt budget — more relevant context in every chat.',
    icon: '🧩',
    date: '2026-10-24',
    tags: ['Knowledge', 'AI'],
  },
  {
    id: 'kanban-3d',
    title: '3D kanban board',
    description:
      'An immersive 3D view of the project board with auto-fitting camera — switch view modes on the Kanban page to try it.',
    icon: '🗂️',
    date: '2026-10-23',
    href: '/kanban',
    tags: ['Page', 'Kanban'],
  },
  {
    id: 'interactive-kanban',
    title: 'Interactive kanban board',
    description:
      'Move cards between columns directly on the board — transitions are validated by the engine, and card notes are hash-chained for tamper evidence.',
    icon: '✅',
    date: '2026-10-22',
    href: '/kanban',
    tags: ['Kanban', 'UX'],
  },
  {
    id: 'fast-boot',
    title: 'Faster startup',
    description:
      'The app boots from a single /bootstrap request and the startup progress overlay stops polling as soon as the server is ready.',
    icon: '🚀',
    date: '2026-10-21',
    tags: ['Performance', 'UX'],
  },
  {
    id: 'chat-transcript',
    title: 'Chat transcript redesign',
    description:
      'The chat transcript now uses a log-entry style layout with a reading gutter, and shows per-turn stream metrics under each response.',
    icon: '📝',
    date: '2026-10-16',
    href: '/chat',
    tags: ['Chat', 'UX'],
  },
  {
    id: 'whats-new-tour',
    title: 'Guided feature tour',
    description:
      'A "What\'s New" button in the bottom-right tracks new features across builds. Open it for a full list, or click "Take guided tour" for a step-by-step walkthrough with spotlight highlights.',
    icon: '✨',
    date: '2026-05-27',
    href: '/',
    tags: ['UI', 'Onboarding'],
  },
  {
    id: 'multimodal-page',
    title: 'Multimodal page',
    description:
      'Dedicated page for vision training, image generation, speech-to-text, text-to-speech, and video processing. Upload images to train the vision model, generate images from prompts, and transcribe audio.',
    icon: '🎨',
    date: '2026-05-27',
    href: '/multimodal',
    tags: ['Page', 'Vision', 'Audio'],
  },
  {
    id: 'conversations-dropdown',
    title: 'Conversations in sidebar',
    description:
      'Chat history accessible from the conversation sidebar — recent messages, starred conversations, all without leaving the chat view.',
    icon: '💬',
    date: '2026-05-27',
    href: '/chat',
    tags: ['UI', 'Navigation'],
  },
  {
    id: 'model-comparison',
    title: 'Model comparison bench',
    description:
      'Side-by-side benchmark results across models. Compare throughput, latency, P95, P99, and memory usage. Best values are highlighted.',
    icon: '📊',
    date: '2026-05-27',
    href: '/benchmark',
    tags: ['Benchmark', 'Models'],
  },
  {
    id: 'model-detail',
    title: 'Model detail page',
    description:
      'Comprehensive per-model view with live uptime, inference count, benchmark metrics, vocabulary size, context length, and load/unload controls.',
    icon: '🔍',
    date: '2026-05-27',
    href: '/model/gpt2',
    tags: ['Page', 'Models'],
  },
  {
    id: 'context-inspector',
    title: 'Context Inspector',
    description:
      'See how the model sees your conversation — trait weights, steering modes, injected knowledge, feedback stats, and workspace memory. In Tools → Context tab.',
    icon: '🧠',
    date: '2026-05-26',
    href: '/chat',
    tags: ['Tool', 'Debug'],
  },
  {
    id: 'live-quality-monitor',
    title: 'Live model quality monitor',
    description:
      'Real-time coherence, quality, and repetition scores alongside system health. Polls every 5 seconds on the monitoring page.',
    icon: '📈',
    date: '2026-05-26',
    href: '/settings',
    tags: ['Monitoring', 'Quality'],
  },
  {
    id: 'model-quality',
    title: 'Model quality card',
    description:
      'Quality KPIs (coherence, repetition, avg length) on the System Health page. See how your model performs over time.',
    icon: '🏆',
    date: '2026-05-26',
    href: '/settings',
    tags: ['Monitoring'],
  },
  {
    id: 'snapshot-weights',
    title: 'Snapshot trait weights',
    description:
      'Save and load soul weight presets from the Souls page. Switch between different trait configurations instantly.',
    icon: '💾',
    date: '2026-05-26',
    href: '/souls',
    tags: ['Personality', 'Persistence'],
  },
  {
    id: 'batch-delete',
    title: 'Batch conversation management',
    description:
      'Select multiple conversations with checkboxes, then delete them all at once. With Select All and inline count display.',
    icon: '🗑️',
    date: '2026-05-27',
    href: '/chat',
    tags: ['Management', 'UX'],
  },
  {
    id: 'sse-standardization',
    title: 'SSE streaming standardization',
    description:
      'All streaming endpoints now use a consistent envelope format: {stream, phase, status, data, meta, message}. More reliable chat & training.',
    icon: '📡',
    date: '2026-05-21',
    tags: ['Infrastructure', 'Reliability'],
  },
]
