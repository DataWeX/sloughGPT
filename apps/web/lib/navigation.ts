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
      { path: '/auto-train', labelKey: 'nav.auto-train', icon: '🤖', description: 'Auto training workflow' },
      { path: '/datasets', labelKey: 'nav.datasets', shortcut: '3', icon: '📊', description: 'Manage datasets' },
      { path: '/dataset', labelKey: 'nav.dataset', icon: '📊', description: 'Dataset detail view' },
      { path: '/knowledge', labelKey: 'nav.knowledge', shortcut: '8', icon: '📚', description: 'Manage knowledge' },
      { path: '/memory', labelKey: 'nav.memory', icon: '🧠', description: 'Memory management' },
    ],
  },
  {
    labelKey: 'nav.section.ai',
    routes: [
      { path: '/models', labelKey: 'nav.models', shortcut: '4', icon: '🧠', description: 'Models & adapters' },
      { path: '/agents', labelKey: 'nav.agents', shortcut: '5', icon: '🤖', description: 'Manage agents' },
      { path: '/companion', labelKey: 'nav.companion', icon: '🧑', description: 'AI personality management' },
      { path: '/multimodal', labelKey: 'nav.multimodal', shortcut: '9', icon: '🎨', description: 'Vision & speech' },
      { path: '/souls', labelKey: 'nav.souls', shortcut: '6', icon: '👻', description: 'Personalities & voice' },
      { path: '/voice', labelKey: 'nav.voice', icon: '🎤', description: 'Voice & TTS settings' },
      { path: '/benchmark', labelKey: 'nav.benchmark', icon: '📈', description: 'Model evaluation' },
      { path: '/compare', labelKey: 'nav.compare', icon: '⚖️', description: 'Compare models' },
      { path: '/tokenizer', labelKey: 'nav.tokenizer', icon: '🔤', description: 'Tokenizer explorer' },
      { path: '/vector', labelKey: 'nav.vector', icon: '📐', description: 'Vector store & embeddings' },
      { path: '/kb', labelKey: 'nav.kb', icon: '📖', description: 'Knowledge base management' },
      { path: '/token-tree', labelKey: 'nav.token-tree', icon: '🌳', description: 'BPE merge tree explorer' },
      { path: '/meta-weights', labelKey: 'nav.meta-weights', icon: '⚖️', description: 'Inference weight tuning' },
      { path: '/infer', labelKey: 'nav.infer', icon: '⚡', description: 'API playground' },
      { path: '/world', labelKey: 'nav.world', icon: '🌍', description: 'World simulation & rendering' },
      { path: '/experiments', labelKey: 'nav.experiments', icon: '🧪', description: 'ML experiment tracking' },
    ],
  },
  {
    labelKey: 'nav.section.system',
    routes: [
      { path: '/monitoring', labelKey: 'nav.monitoring', shortcut: '7', icon: '💓', description: 'System status' },
      { path: '/errors', labelKey: 'nav.errors', icon: '🔴', description: 'Client error monitoring' },
      { path: '/security', labelKey: 'nav.security', icon: '🛡️', description: 'Security audit logs' },
      { path: '/auth', labelKey: 'nav.auth', icon: '🔐', description: 'Authentication' },
      { path: '/settings', labelKey: 'nav.settings', shortcut: 'shift+A', icon: '⚙️', description: 'App settings' },
      { path: '/export', labelKey: 'nav.export', icon: '📦', description: 'Export models & data' },
      { path: '/files', labelKey: 'nav.files', icon: '📁', description: 'File management' },
      { path: '/images', labelKey: 'nav.images', icon: '🖼️', description: 'Image generation' },
      { path: '/registry', labelKey: 'nav.registry', icon: '📋', description: 'Model registry' },
      { path: '/vm', labelKey: 'nav.vm', icon: '💻', description: 'VM sandbox console' },
      { path: '/session', labelKey: 'nav.session', icon: '🔍', description: 'Session inspector' },
      { path: '/shell', labelKey: 'nav.shell', icon: '⌨️', description: 'Dait shell terminal' },
    ],
  },
  {
    labelKey: 'nav.section.tools',
    routes: [
      { path: '/feedback', labelKey: 'nav.feedback', icon: '💬', description: 'Feedback analytics' },
      { path: '/adapters', labelKey: 'nav.adapters', icon: '🔧', description: 'LoRA adapter management' },
      { path: '/evaluate', labelKey: 'nav.evaluate', icon: '📊', description: 'Model evaluation' },
      { path: '/learn', labelKey: 'nav.learn', icon: '📖', description: 'Learning pipeline' },
      { path: '/self-train', labelKey: 'nav.self-train', icon: '🏋️', description: 'Self-training loop' },
      { path: '/collections', labelKey: 'nav.collections', icon: '📂', description: 'Data collections' },
      { path: '/docstore', labelKey: 'nav.docstore', icon: '📄', description: 'Document store' },
      { path: '/workflow', labelKey: 'nav.workflow', icon: '🔄', description: 'Feedback pipeline' },
      { path: '/admin', labelKey: 'nav.admin', icon: '🛠️', description: 'Administration' },
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
  '/datasets': IconBrain,
  '/dataset': IconBrain,
  '/knowledge': IconSearch,
  '/memory': IconBrain,
  '/models': IconModels,
  '/agents': IconAgents,
  '/companion': IconBrain,
  '/multimodal': IconVision,
  '/souls': IconBrain,
  '/voice': IconChat,
  '/benchmark': IconActivity,
  '/compare': IconBrain,
  '/tokenizer': IconBrain,
  '/vector': IconBrain,
  '/meta-weights': IconActivity,
  '/infer': IconBrain,
  '/world': IconVision,
  '/kb': IconBrain,
  '/token-tree': IconBrain,
  '/experiments': IconActivity,
  '/monitoring': IconActivity,
  '/errors': IconActivity,
  '/security': IconSettings,
  '/auth': IconSettings,
  '/settings': IconSettings,
  '/export': IconDownload,
  '/files': IconBrain,
  '/images': IconVision,
  '/registry': IconBrain,
  '/vm': IconBrain,
  '/session': IconSearch,
  '/shell': IconActivity,
  '/feedback': IconActivity,
  '/adapters': IconSettings,
  '/evaluate': IconActivity,
  '/learn': IconBrain,
  '/self-train': IconTraining,
  '/collections': IconBrain,
  '/docstore': IconBrain,
  '/workflow': IconActivity,
  '/admin': IconSettings,
}
