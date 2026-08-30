'use client'

import {
  IconChat,
  IconModels,
  IconSettings,
  IconTraining,
  IconBrain,
  IconAgents,
  IconSearch,
  IconActivity,
  IconTokenizer,
  IconDocument,
  IconCode,
  IconChart,
  IconBookmark,
  IconSparkle,
  IconBeaker,
  IconBenchmark,
  IconMapPin,
  IconError,
  IconMic,
  IconThumbUp,
  IconGrid,
  IconLabs,
  IconAlert,
} from '@/components/icons/NavIcons'
import type { ComponentType } from 'react'

/* ------------------------------------------------------------------ */
/*  Single source of truth for all navigation routes, sections,       */
/*  keyboard shortcuts, and command palette metadata.                  */
/* ------------------------------------------------------------------ */

export interface NavRoute {
  path: string
  labelKey: string
  shortcut?: string
  icon?: string
  description?: string
}

export interface NavSectionDef {
  labelKey: string
  routes: NavRoute[]
}

/** Sidebar sections + command palette grouping + keyboard shortcut labels */
export const NAV_SECTIONS: NavSectionDef[] = [
  {
    labelKey: 'nav.section.core',
    routes: [
      { path: '/chat', labelKey: 'nav.chat', shortcut: '1', icon: 'chat', description: 'Open chat' },
      { path: '/training', labelKey: 'nav.training', shortcut: '2', icon: 'training', description: 'Train models' },
      { path: '/auto-train', labelKey: 'nav.auto-train', icon: 'brain', description: 'Auto training workflow' },
      { path: '/datasets', labelKey: 'nav.datasets', shortcut: '3', icon: 'chart', description: 'Manage datasets' },
      { path: '/session', labelKey: 'nav.session', icon: 'search', description: 'Inspect & debug sessions' },
      { path: '/knowledge', labelKey: 'nav.knowledge', shortcut: '8', icon: 'search', description: 'Knowledge & memory' },
    ],
  },
  {
    labelKey: 'nav.section.ai',
    routes: [
      { path: '/models', labelKey: 'nav.models', shortcut: '4', icon: 'models', description: 'Models, adapters & inference' },
      { path: '/agents', labelKey: 'nav.agents', shortcut: '5', icon: 'agents', description: 'Manage agents' },
      { path: '/souls', labelKey: 'nav.souls', shortcut: '6', icon: 'brain', description: 'Personalities & voice' },
      { path: '/experiments', labelKey: 'nav.experiments', icon: 'activity', description: 'Experiment tracking' },
      { path: '/benchmark', labelKey: 'nav.benchmark', icon: 'activity', description: 'Evaluation & comparison' },
      { path: '/tokenizer', labelKey: 'nav.tokenizer', icon: 'tokenizer', description: 'Tokenizer tools' },
      { path: '/vector', labelKey: 'nav.vector', icon: 'brain', description: 'Vector store & embeddings' },
    ],
  },
  {
    labelKey: 'nav.section.system',
    routes: [
      { path: '/monitoring', labelKey: 'nav.monitoring', shortcut: '7', icon: 'activity', description: 'System status' },
      { path: '/errors', labelKey: 'nav.errors', icon: 'error', description: 'Client error monitoring' },
      { path: '/security', labelKey: 'nav.security', icon: 'settings', description: 'Security audit logs' },
      { path: '/settings', labelKey: 'nav.settings', shortcut: 'shift+A', icon: 'settings', description: 'App settings' },
      { path: '/files', labelKey: 'nav.files', icon: 'file', description: 'Files & images' },
      { path: '/voice', labelKey: 'nav.voice', icon: 'chat', description: 'Voice & audio settings' },
      { path: '/shell', labelKey: 'nav.shell', icon: 'code', description: 'Shell & debugging' },
    ],
  },
  {
    labelKey: 'nav.section.tools',
    routes: [
      { path: '/feedback', labelKey: 'nav.feedback', icon: 'activity', description: 'Feedback analytics' },
      { path: '/kanban', labelKey: 'nav.kanban', icon: 'kanban', description: 'Project kanban board' },
      { path: '/adapters', labelKey: 'nav.adapters', icon: 'settings', description: 'LoRA adapter management' },
    ],
  },
]

/** Flat route map for shortcut lookup: shortcut → path */
export const SHORTCUT_TO_PATH: Record<string, string> = {}
for (const section of NAV_SECTIONS) {
  for (const route of section.routes) {
    if (route.shortcut) SHORTCUT_TO_PATH[route.shortcut] = route.path
  }
}

/** Flat list for command palette (preserves section order) */
export const ALL_ROUTES: NavRoute[] = NAV_SECTIONS.flatMap(s => s.routes)

/** Sidebar icon map — keyed by path */
export const SIDEBAR_ICONS: Record<string, ComponentType<{ className?: string }>> = {
  '/chat': IconChat,
  '/training': IconTraining,
  '/auto-train': IconBrain,
  '/datasets': IconChart,
  '/session': IconSearch,
  '/knowledge': IconBookmark,
  '/models': IconModels,
  '/agents': IconAgents,
  '/souls': IconSparkle,
  '/experiments': IconBeaker,
  '/benchmark': IconBenchmark,
  '/tokenizer': IconTokenizer,
  '/vector': IconMapPin,
  '/monitoring': IconActivity,
  '/errors': IconError,
  '/security': IconAlert,
  '/settings': IconSettings,
  '/files': IconDocument,
  '/voice': IconMic,
  '/shell': IconCode,
  '/feedback': IconThumbUp,
  '/kanban': IconGrid,
  '/adapters': IconLabs,
}
