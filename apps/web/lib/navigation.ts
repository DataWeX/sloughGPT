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
      { path: '/chat', labelKey: 'nav.chat', shortcut: '1', icon: 'chat', description: 'Open chat' },
      { path: '/training', labelKey: 'nav.training', shortcut: '2', icon: 'training', description: 'Train models' },
      { path: '/auto-train', labelKey: 'nav.auto-train', icon: 'brain', description: 'Auto training workflow' },
      { path: '/datasets', labelKey: 'nav.datasets', shortcut: '3', icon: 'chart', description: 'Manage datasets' },
      { path: '/dataset', labelKey: 'nav.dataset', icon: 'chart', description: 'Dataset detail view' },
      { path: '/knowledge', labelKey: 'nav.knowledge', shortcut: '8', icon: 'search', description: 'Manage knowledge' },
      { path: '/memory', labelKey: 'nav.memory', icon: 'brain', description: 'Memory management' },
    ],
  },
  {
    labelKey: 'nav.section.ai',
    routes: [
      { path: '/models', labelKey: 'nav.models', shortcut: '4', icon: 'models', description: 'Models & adapters' },
      { path: '/agents', labelKey: 'nav.agents', shortcut: '5', icon: 'agents', description: 'Manage agents' },
      { path: '/companion', labelKey: 'nav.companion', icon: 'brain', description: 'AI personality management' },
      { path: '/multimodal', labelKey: 'nav.multimodal', shortcut: '9', icon: 'vision', description: 'Vision & speech' },
      { path: '/souls', labelKey: 'nav.souls', shortcut: '6', icon: 'brain', description: 'Personalities & voice' },
      { path: '/voice', labelKey: 'nav.voice', icon: 'chat', description: 'Voice & TTS settings' },
      { path: '/benchmark', labelKey: 'nav.benchmark', icon: 'activity', description: 'Model evaluation' },
      { path: '/compare', labelKey: 'nav.compare', icon: 'compare', description: 'Compare models' },
      { path: '/tokenizer', labelKey: 'nav.tokenizer', icon: 'tokenizer', description: 'Tokenizer explorer' },
      { path: '/vector', labelKey: 'nav.vector', icon: 'brain', description: 'Vector store & embeddings' },
      { path: '/kb', labelKey: 'nav.kb', icon: 'brain', description: 'Knowledge base management' },
      { path: '/token-tree', labelKey: 'nav.token-tree', icon: 'brain', description: 'BPE merge tree explorer' },
      { path: '/meta-weights', labelKey: 'nav.meta-weights', icon: 'activity', description: 'Inference weight tuning' },
      { path: '/infer', labelKey: 'nav.infer', icon: 'bolt', description: 'API playground' },
      { path: '/world', labelKey: 'nav.world', icon: 'vision', description: 'World simulation & rendering' },
      { path: '/experiments', labelKey: 'nav.experiments', icon: 'beaker', description: 'ML experiment tracking' },
    ],
  },
  {
    labelKey: 'nav.section.system',
    routes: [
      { path: '/monitoring', labelKey: 'nav.monitoring', shortcut: '7', icon: 'activity', description: 'System status' },
      { path: '/errors', labelKey: 'nav.errors', icon: 'error', description: 'Client error monitoring' },
      { path: '/security', labelKey: 'nav.security', icon: 'settings', description: 'Security audit logs' },
      { path: '/auth', labelKey: 'nav.auth', icon: 'settings', description: 'Authentication' },
      { path: '/settings', labelKey: 'nav.settings', shortcut: 'shift+A', icon: 'settings', description: 'App settings' },
      { path: '/export', labelKey: 'nav.export', icon: 'export', description: 'Export models & data' },
      { path: '/files', labelKey: 'nav.files', icon: 'file', description: 'File management' },
      { path: '/images', labelKey: 'nav.images', icon: 'vision', description: 'Image generation' },
      { path: '/registry', labelKey: 'nav.registry', icon: 'brain', description: 'Model registry' },
      { path: '/vm', labelKey: 'nav.vm', icon: 'brain', description: 'VM sandbox console' },
      { path: '/session', labelKey: 'nav.session', icon: 'search', description: 'Session inspector' },
      { path: '/shell', labelKey: 'nav.shell', icon: 'code', description: 'Dait shell terminal' },
    ],
  },
  {
    labelKey: 'nav.section.tools',
    routes: [
      { path: '/feedback', labelKey: 'nav.feedback', icon: 'activity', description: 'Feedback analytics' },
      { path: '/adapters', labelKey: 'nav.adapters', icon: 'settings', description: 'LoRA adapter management' },
      { path: '/lora-eval', labelKey: 'nav.lora-eval', icon: 'beaker', description: 'LoRA adapter evaluation' },
      { path: '/rate-limit', labelKey: 'nav.rate-limit', icon: 'activity', description: 'Rate limiting status' },
      { path: '/evaluate', labelKey: 'nav.evaluate', icon: 'chart', description: 'Model evaluation' },
      { path: '/learn', labelKey: 'nav.learn', icon: 'brain', description: 'Learning pipeline' },
      { path: '/self-train', labelKey: 'nav.self-train', icon: 'training', description: 'Self-training loop' },
      { path: '/collections', labelKey: 'nav.collections', icon: 'brain', description: 'Data collections' },
      { path: '/docstore', labelKey: 'nav.docstore', icon: 'file', description: 'Document store' },
      { path: '/workflow', labelKey: 'nav.workflow', icon: 'activity', description: 'Feedback pipeline' },
      { path: '/admin', labelKey: 'nav.admin', icon: 'settings', description: 'Administration' },
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
  '/lora-eval': IconActivity,
  '/rate-limit': IconSettings,
  '/evaluate': IconActivity,
  '/learn': IconBrain,
  '/self-train': IconTraining,
  '/collections': IconBrain,
  '/docstore': IconBrain,
  '/workflow': IconActivity,
  '/admin': IconSettings,
}
