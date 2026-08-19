'use client'

import {
  IconChat,
  IconModels,
  IconSettings,
  IconTraining,
  IconBrain,
  IconAgents,
  IconVision,
  IconSearch,
  IconActivity,
  IconDownload,
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
      { path: '/chat', labelKey: 'nav.chat', shortcut: '1', icon: '💬', description: 'Open chat' },
      { path: '/training', labelKey: 'nav.training', shortcut: '2', icon: '🏋️', description: 'Train models' },
      { path: '/datasets', labelKey: 'nav.datasets', shortcut: '3', icon: '📊', description: 'Manage datasets' },
      { path: '/knowledge', labelKey: 'nav.knowledge', shortcut: '8', icon: '📚', description: 'Manage knowledge' },
    ],
  },
  {
    labelKey: 'nav.section.ai',
    routes: [
      { path: '/models', labelKey: 'nav.models', shortcut: '4', icon: '🧠', description: 'Models & adapters' },
      { path: '/agents', labelKey: 'nav.agents', shortcut: '5', icon: '🤖', description: 'Manage agents' },
      { path: '/multimodal', labelKey: 'nav.multimodal', shortcut: '9', icon: '🎨', description: 'Vision & speech' },
      { path: '/souls', labelKey: 'nav.souls', shortcut: '6', icon: '👻', description: 'Personalities & voice' },
      { path: '/benchmark', labelKey: 'nav.benchmark', icon: '📈', description: 'Model evaluation' },
      { path: '/compare', labelKey: 'nav.compare', icon: '⚖️', description: 'Compare models' },
      { path: '/tokenizer', labelKey: 'nav.tokenizer', icon: '🔤', description: 'Tokenizer explorer' },
    ],
  },
  {
    labelKey: 'nav.section.system',
    routes: [
      { path: '/monitoring', labelKey: 'nav.monitoring', shortcut: '7', icon: '💓', description: 'System status' },
      { path: '/settings', labelKey: 'nav.settings', shortcut: 'shift+A', icon: '⚙️', description: 'App settings' },
      { path: '/export', labelKey: 'nav.export', icon: '📦', description: 'Export models & data' },
      { path: '/files', labelKey: 'nav.files', icon: '📁', description: 'File management' },
      { path: '/images', labelKey: 'nav.images', icon: '🖼️', description: 'Image generation' },
    ],
  },
  {
    labelKey: 'nav.section.tools',
    routes: [
      { path: '/feedback', labelKey: 'nav.feedback', icon: '💬', description: 'Feedback analytics' },
      { path: '/adapters', labelKey: 'nav.adapters', icon: '🔧', description: 'LoRA adapter management' },
      { path: '/learn', labelKey: 'nav.learn', icon: '📖', description: 'Learning pipeline' },
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
  '/datasets': IconBrain,
  '/knowledge': IconSearch,
  '/models': IconModels,
  '/agents': IconAgents,
  '/multimodal': IconVision,
  '/souls': IconBrain,
  '/benchmark': IconActivity,
  '/compare': IconBrain,
  '/tokenizer': IconBrain,
  '/monitoring': IconActivity,
  '/settings': IconSettings,
  '/export': IconDownload,
  '/files': IconBrain,
  '/images': IconVision,
  '/feedback': IconActivity,
  '/adapters': IconSettings,
  '/learn': IconBrain,
}
