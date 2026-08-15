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
  IconCompare,
  IconActivity,
  IconDownload,
  IconBenchmark,
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
      { path: '/models', labelKey: 'nav.models', shortcut: '4', icon: '🧠', description: 'Manage personalities' },
      { path: '/agents', labelKey: 'nav.agents', shortcut: '5', icon: '🤖', description: 'Manage agents' },
      { path: '/multimodal', labelKey: 'nav.multimodal', shortcut: '9', icon: '🎨', description: 'Vision & speech' },
      { path: '/compare', labelKey: 'nav.compare', shortcut: '6', icon: '⚖️', description: 'Compare models' },
      { path: '/souls', labelKey: 'nav.souls', icon: '👻', description: 'Personality management' },
    ],
  },
  {
    labelKey: 'nav.section.system',
    routes: [
      { path: '/monitoring', labelKey: 'nav.monitoring', shortcut: '7', icon: '💓', description: 'System status' },
      { path: '/benchmark', labelKey: 'nav.benchmark', icon: '📏', description: 'Model evaluation' },
      { path: '/tokenizer', labelKey: 'nav.tokenizer', icon: '🔤', description: 'BPE tokenizer' },
      { path: '/settings', labelKey: 'nav.settings', shortcut: 'shift+A', icon: '⚙️', description: 'App settings' },
    ],
  },
  {
    labelKey: 'nav.section.tools',
    routes: [
      { path: '/adapters', labelKey: 'nav.adapters', icon: '🔧', description: 'LoRA adapter management' },
      { path: '/feedback', labelKey: 'nav.feedback', icon: '💬', description: 'Feedback analytics' },
      { path: '/companion', labelKey: 'nav.companion', icon: '🧠', description: 'AI personality' },
      { path: '/voice', labelKey: 'nav.voice', icon: '🔊', description: 'Text-to-speech' },
      { path: '/learn', labelKey: 'nav.learn', icon: '🔍', description: 'Continual web learning' },
      { path: '/export', labelKey: 'nav.export', icon: '📦', description: 'Export models & data' },
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
  '/compare': IconCompare,
  '/souls': IconBrain,
  '/monitoring': IconActivity,
  '/benchmark': IconBenchmark,
  '/tokenizer': IconSettings,
  '/settings': IconSettings,
  '/adapters': IconSettings,
  '/feedback': IconActivity,
  '/companion': IconBrain,
  '/voice': IconActivity,
  '/learn': IconSearch,
  '/export': IconDownload,
}
