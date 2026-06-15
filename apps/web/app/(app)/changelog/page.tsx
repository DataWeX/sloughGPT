'use client'

import { useEffect } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui'

const CHANGELOG_KEY = 'changelog_seen_version'
const CURRENT_VERSION = '2026-06-15'

export function hasUnseenChangelog(): boolean {
  if (typeof window === 'undefined') return false
  try {
    const seen = localStorage.getItem(CHANGELOG_KEY)
    return seen !== CURRENT_VERSION
  } catch { return false }
}

export function markChangelogSeen(): void {
  try { localStorage.setItem(CHANGELOG_KEY, CURRENT_VERSION) } catch {}
}

interface ChangelogEntry {
  date: string
  title: string
  items: { text: string; type: 'feature' | 'fix' | 'improvement' }[]
}

const ENTRIES: ChangelogEntry[] = [
  {
    date: 'June 2026',
    title: 'System Status & Quick Actions',
    items: [
      { text: 'System status card on home page — see API, model, soul, and knowledge at a glance', type: 'feature' },
      { text: 'Chat health indicator — green/amber/red dot showing backend status', type: 'feature' },
      { text: 'Continue last conversation from home page', type: 'feature' },
      { text: 'Dataset preview — see first samples before training', type: 'feature' },
      { text: 'Auto-scroll loss chart during training (40-point sliding window)', type: 'improvement' },
      { text: 'Dataset "Train" quick action — one click to start training', type: 'feature' },
      { text: 'Onboarding hint for first-time users', type: 'feature' },
    ],
  },
  {
    date: 'May 2026',
    title: 'Training & personas',
    items: [
      { text: 'Quick Train mode — pick a dataset, hit train, auto-config handles the rest', type: 'feature' },
      { text: 'Plain-language status at every touchpoint — no ML jargon', type: 'improvement' },
      { text: 'Dataset stats with format auto-detection and suggested training method', type: 'feature' },
      { text: 'Training job detail page with loss chart and reward KPIs', type: 'feature' },
      { text: 'Checkpoint comparison dialog', type: 'feature' },
      { text: 'Datasets management page with search, batch delete, export', type: 'feature' },
      { text: 'Model descriptions in chat dropdown', type: 'feature' },
      { text: 'Chat-to-dataset export — save conversations as training data', type: 'feature' },
      { text: 'Home page training status widget', type: 'feature' },
    ],
  },
  {
    date: 'May 2026',
    title: 'Chat & models',
    items: [
      { text: 'Model health indicator in chat header', type: 'feature' },
      { text: 'Knowledge panel in chat — inject facts into conversations', type: 'feature' },
      { text: 'Soul hot-reload — switch personalities without model restart', type: 'feature' },
      { text: 'Checkpoint loading from model catalog', type: 'feature' },
      { text: 'Markdown rendering in chat (code blocks, headings, lists)', type: 'feature' },
      { text: 'Voice input and image upload (pending multimodal inference)', type: 'feature' },
      { text: 'Mobile-responsive chat layout', type: 'improvement' },
    ],
  },
  {
    date: 'April 2026',
    title: 'Infrastructure',
    items: [
      { text: 'SloNet — pure NumPy autograd engine (no PyTorch dependency)', type: 'feature' },
      { text: 'Model server with circuit breaker and MPS OOM recovery', type: 'feature' },
      { text: 'Unified KV cache for inference optimization', type: 'feature' },
      { text: 'Shell OS with 40+ commands, pipelines, and tab completion', type: 'feature' },
      { text: 'Multi-agent orchestration', type: 'feature' },
      { text: 'Context manager architecture with trait weights', type: 'feature' },
    ],
  },
]

const TYPE_STYLES = {
  feature: 'bg-primary/10 text-primary',
  fix: 'bg-amber-500/10 text-amber-600',
  improvement: 'bg-emerald-500/10 text-emerald-600',
}

export default function ChangelogPage() {
  useEffect(() => { markChangelogSeen() }, [])

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="What's new" subtitle="Recent features and improvements" />} />

      <div className="space-y-6">
        {ENTRIES.map((entry, i) => (
          <div key={i}>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xs font-medium text-muted-foreground">{entry.date}</span>
              <h2 className="text-sm font-semibold">{entry.title}</h2>
            </div>
            <div className="space-y-1.5">
              {entry.items.map((item, j) => (
                <div key={j} className="flex items-start gap-2 text-xs">
                  <Badge
                    label={item.type}
                    className={`mt-0.5 shrink-0 text-[9px] px-1.5 py-0 ${TYPE_STYLES[item.type]}`}
                  />
                  <span className="text-muted-foreground leading-relaxed">{item.text}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
