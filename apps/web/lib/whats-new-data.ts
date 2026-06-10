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
    id: 'whats-new-tour',
    title: 'Guided feature tour',
    description: 'A "What\'s New" button in the bottom-right tracks new features across builds. Open it for a full list, or click "Take guided tour" for a step-by-step walkthrough with spotlight highlights.',
    icon: '✨',
    date: '2026-05-27',
    href: '/',
    tags: ['UI', 'Onboarding'],
  },
  {
    id: 'multimodal-page',
    title: 'Multimodal page',
    description: 'Dedicated page for vision training, image generation, speech-to-text, text-to-speech, and video processing. Upload images to train the vision model, generate images from prompts, and transcribe audio.',
    icon: '🎨',
    date: '2026-05-27',
    href: '/multimodal',
    tags: ['Page', 'Vision', 'Audio'],
  },
  {
    id: 'conversations-dropdown',
    title: 'Conversations in navbar',
    description: 'Chat sidebar moved to a dropdown in the header — recent messages, starred conversations, and a link to view all, without taking up screen space.',
    icon: '💬',
    date: '2026-05-27',
    href: '/chat',
    tags: ['UI', 'Navigation'],
  },
  {
    id: 'conversations-page',
    title: 'Conversations page',
    description: 'Full conversation browser with search, star/pin/rename, export (MD/JSON), and batch select & delete. Access from sidebar or chat header.',
    icon: '📋',
    date: '2026-05-27',
    href: '/conversations',
    tags: ['Page', 'Management'],
  },
  {
    id: 'model-comparison',
    title: 'Model comparison bench',
    description: 'Side-by-side benchmark results across models. Compare throughput, latency, P95, P99, and memory usage. Best values are highlighted.',
    icon: '📊',
    date: '2026-05-27',
    href: '/compare',
    tags: ['Benchmark', 'Models'],
  },
  {
    id: 'model-detail',
    title: 'Model detail page',
    description: 'Comprehensive per-model view with live uptime, inference count, benchmark metrics, vocabulary size, context length, and load/unload controls.',
    icon: '🔍',
    date: '2026-05-27',
    href: '/model/gpt2',
    tags: ['Page', 'Models'],
  },
  {
    id: 'context-inspector',
    title: 'Context Inspector',
    description: 'See how the model sees your conversation — trait weights, steering modes, injected knowledge, feedback stats, and workspace memory. In Tools → Context tab.',
    icon: '🧠',
    date: '2026-05-26',
    href: '/chat',
    tags: ['Tool', 'Debug'],
  },
  {
    id: 'live-quality-monitor',
    title: 'Live model quality monitor',
    description: 'Real-time coherence, quality, and repetition scores alongside system health. Polls every 5 seconds on the monitoring page.',
    icon: '📈',
    date: '2026-05-26',
    href: '/monitoring',
    tags: ['Monitoring', 'Quality'],
  },
  {
    id: 'model-quality',
    title: 'Model quality card',
    description: 'Quality KPIs (coherence, repetition, avg length) on the System Health page. See how your model performs over time.',
    icon: '🏆',
    date: '2026-05-26',
    href: '/monitoring',
    tags: ['Monitoring'],
  },
  {
    id: 'snapshot-weights',
    title: 'Snapshot trait weights',
    description: 'Save and load personality weight presets from the models page. Switch between different trait configurations instantly.',
    icon: '💾',
    date: '2026-05-26',
    href: '/models',
    tags: ['Personality', 'Persistence'],
  },
  {
    id: 'batch-delete',
    title: 'Batch conversation management',
    description: 'Select multiple conversations with checkboxes, then delete them all at once. With Select All and inline count display.',
    icon: '🗑️',
    date: '2026-05-27',
    href: '/conversations',
    tags: ['Management', 'UX'],
  },
  {
    id: 'sse-standardization',
    title: 'SSE streaming standardization',
    description: 'All streaming endpoints now use a consistent envelope format: {stream, phase, status, data, meta, message}. More reliable chat & training.',
    icon: '📡',
    date: '2026-05-21',
    tags: ['Infrastructure', 'Reliability'],
  },
]
