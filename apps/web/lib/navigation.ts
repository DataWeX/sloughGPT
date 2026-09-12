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
  IconUsers,
  IconClock,
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
      { path: '/training/trends', labelKey: 'nav.training_trends', icon: 'chart', description: 'Training quality trends' },
      { path: '/training/insights', labelKey: 'nav.training_insights', icon: 'sparkle', description: 'Adaptive learning intelligence' },
      { path: '/training/grid-search', labelKey: 'nav.training_grid_search', icon: 'grid', description: 'Hyperparameter optimization' },
      { path: '/training/model-card', labelKey: 'nav.training_model_card', icon: 'file', description: 'Generate model cards' },
      { path: '/training/compare', labelKey: 'nav.training_compare', icon: 'compare', description: 'Compare training runs' },
      { path: '/training/presets', labelKey: 'nav.training_presets', icon: 'zap', description: 'Quick-start training templates' },
      { path: '/training/runs', labelKey: 'nav.training_runs', icon: 'clock', description: 'View all training runs' },
      { path: '/training/analytics', labelKey: 'nav.training_analytics', icon: 'chart', description: 'Training analytics dashboard' },
      { path: '/auto-train', labelKey: 'nav.auto_train', icon: 'zap', description: 'Auto-training from conversations' },
      { path: '/datasets', labelKey: 'nav.datasets', shortcut: '3', icon: 'chart', description: 'Manage datasets' },
      { path: '/knowledge', labelKey: 'nav.knowledge', shortcut: '8', icon: 'search', description: 'Knowledge & memory' },
    ],
  },
  {
    labelKey: 'nav.section.ai',
    routes: [
      { path: '/models', labelKey: 'nav.models', shortcut: '4', icon: 'models', description: 'Models, adapters & inference' },
      { path: '/agents', labelKey: 'nav.agents', shortcut: '5', icon: 'agents', description: 'Manage agents' },
      { path: '/souls', labelKey: 'nav.souls', shortcut: '6', icon: 'brain', description: 'Personalities & voice' },
      { path: '/personality', labelKey: 'nav.personality', icon: 'brain', description: 'Personality & voice' },
      { path: '/consciousness/quickstart', labelKey: 'nav.consciousness_quickstart', icon: 'sparkle', description: 'Consciousness quick start guide' },
      { path: '/consciousness', labelKey: 'nav.consciousness', icon: 'sparkle', description: 'Consciousness system' },
      { path: '/consciousness/dashboard', labelKey: 'nav.consciousness_dashboard', icon: 'chart', description: 'Consciousness evolution dashboard' },
      { path: '/consciousness/master', labelKey: 'nav.consciousness_master', icon: 'grid', description: 'Consciousness master dashboard' },
      { path: '/consciousness/training', labelKey: 'nav.consciousness_training', icon: 'training', description: 'Consciousness training' },
      { path: '/consciousness/playground', labelKey: 'nav.consciousness_playground', icon: 'sparkle', description: 'Interactive consciousness playground' },
      { path: '/consciousness/analytics', labelKey: 'nav.consciousness_analytics', icon: 'chart', description: 'Consciousness analytics & trends' },
      { path: '/consciousness/statistics', labelKey: 'nav.consciousness_statistics', icon: 'chart', description: 'Consciousness statistics dashboard' },
      { path: '/consciousness/history', labelKey: 'nav.consciousness_history', icon: 'clock', description: 'Consciousness episode history' },
      { path: '/consciousness/personality', labelKey: 'nav.consciousness_personality', icon: 'brain', description: 'Consciousness personality' },
      { path: '/consciousness/settings', labelKey: 'nav.consciousness_settings', icon: 'settings', description: 'Consciousness settings' },
      { path: '/consciousness/all-settings', labelKey: 'nav.consciousness_all_settings', icon: 'settings', description: 'All consciousness settings' },
      { path: '/consciousness/testing', labelKey: 'nav.consciousness_testing', icon: 'beaker', description: 'Consciousness testing & debugging' },
      { path: '/consciousness/test-runner', labelKey: 'nav.consciousness_test_runner', icon: 'beaker', description: 'Automated consciousness test runner' },
      { path: '/consciousness/benchmark', labelKey: 'nav.consciousness_benchmark', icon: 'activity', description: 'Performance benchmark for consciousness system' },
      { path: '/consciousness/docs', labelKey: 'nav.consciousness_docs', icon: 'code', description: 'Consciousness API documentation' },
      { path: '/consciousness/api-explorer', labelKey: 'nav.consciousness_api_explorer', icon: 'code', description: 'Interactive consciousness API explorer' },
      { path: '/consciousness/debug', labelKey: 'nav.consciousness_debug', icon: 'code', description: 'Consciousness system debugging' },
      { path: '/consciousness/monitor', labelKey: 'nav.consciousness_monitor', icon: 'activity', description: 'Real-time consciousness monitoring' },
      { path: '/consciousness/insights', labelKey: 'nav.consciousness_insights', icon: 'sparkle', description: 'AI-powered consciousness insights' },
      { path: '/consciousness/help', labelKey: 'nav.consciousness_help', icon: 'bookmark', description: 'Consciousness help & FAQ' },
      { path: '/consciousness/compare', labelKey: 'nav.consciousness_compare', icon: 'chart', description: 'Compare consciousness configurations' },
      { path: '/consciousness/alerts', labelKey: 'nav.consciousness_alerts', icon: 'activity', description: 'Consciousness alert rules' },
      { path: '/consciousness/export', labelKey: 'nav.consciousness_export', icon: 'code', description: 'Export consciousness data' },
      { path: '/consciousness/health', labelKey: 'nav.consciousness_health', icon: 'activity', description: 'Consciousness system health' },
      { path: '/consciousness/versions', labelKey: 'nav.consciousness_versions', icon: 'clock', description: 'Consciousness version history' },
      { path: '/benchmark', labelKey: 'nav.benchmark', icon: 'activity', description: 'Evaluation & comparison' },
    ],
  },
  {
    labelKey: 'nav.section.system',
    routes: [
      { path: '/monitoring', labelKey: 'nav.monitoring', shortcut: '7', icon: 'activity', description: 'System status & errors' },
      { path: '/settings', labelKey: 'nav.settings', shortcut: 'shift+A', icon: 'settings', description: 'App settings' },
      { path: '/developer', labelKey: 'nav.developer', icon: 'code', description: 'Files, voice & shell' },
    ],
  },
  {
    labelKey: 'nav.section.tools',
    routes: [
      { path: '/writing', labelKey: 'nav.writing', icon: 'document', description: 'Writing assistant' },
      { path: '/translate', labelKey: 'nav.translate', icon: 'chat', description: 'Translate text' },
      { path: '/rewrite', labelKey: 'nav.rewrite', icon: 'sparkle', description: 'Rewrite & polish text' },
      { path: '/brainstorm', labelKey: 'nav.brainstorm', icon: 'brain', description: 'Brainstorm ideas' },
      { path: '/decide', labelKey: 'nav.decide', icon: 'chart', description: 'Help me decide' },
      { path: '/explain', labelKey: 'nav.explain', icon: 'search', description: 'Explain things simply' },
      { path: '/wellness', labelKey: 'nav.wellness', icon: 'sparkle', description: 'Wellness & relaxation' },
      { path: '/feedback', labelKey: 'nav.feedback', icon: 'activity', description: 'Feedback analytics' },
      { path: '/planner', labelKey: 'nav.planner', icon: 'grid', description: 'Planner board + notes' },
      { path: '/phoneme', labelKey: 'nav.phoneme', icon: 'tokenizer', description: 'Phoneme encoding & pronunciation' },
      { path: '/plugins-cloud', labelKey: 'nav.plugins_cloud', icon: 'settings', description: 'Plugins & cloud training' },
    ],
  },
  {
    labelKey: 'nav.section.admin',
    routes: [
      { path: '/workspace-dashboard', labelKey: 'nav.workspace_dashboard', icon: 'activity', description: 'Workspace overview' },
      { path: '/members', labelKey: 'nav.members', icon: 'agents', description: 'Manage members' },
      { path: '/usage', labelKey: 'nav.usage', icon: 'chart', description: 'Usage metrics' },
      { path: '/audit-trail', labelKey: 'nav.audit_trail', icon: 'activity', description: 'Audit log' },
      { path: '/workspace-settings', labelKey: 'nav.workspace_settings', icon: 'settings', description: 'Workspace settings' },
      { path: '/api-keys', labelKey: 'nav.api_keys', icon: 'code', description: 'API key management' },
      { path: '/notifications', labelKey: 'nav.notifications', icon: 'activity', description: 'Workspace notifications' },
      { path: '/workspace-search', labelKey: 'nav.workspace_search', icon: 'search', description: 'Search workspace data' },
      { path: '/permissions', labelKey: 'nav.permissions', icon: 'settings', description: 'Role permissions' },
      { path: '/shared-data', labelKey: 'nav.shared_data', icon: 'users', description: 'Cross-workspace data sharing' },
      { path: '/shortcuts', labelKey: 'nav.shortcuts', icon: 'settings', description: 'Keyboard shortcuts reference' },
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
  '/training/trends': IconChart,
  '/training/insights': IconSparkle,
  '/training/grid-search': IconGrid,
  '/training/model-card': IconDocument,
  '/training/compare': IconChart,
  '/training/presets': IconBeaker,
  '/training/runs': IconClock,
  '/training/analytics': IconChart,
  '/auto-train': IconSparkle,
  '/datasets': IconChart,
  '/knowledge': IconBookmark,
  '/models': IconModels,
  '/agents': IconAgents,
  '/souls': IconSparkle,
  '/personality': IconBrain,
  '/consciousness': IconSparkle,
  '/consciousness/quickstart': IconSparkle,
  '/consciousness/dashboard': IconChart,
  '/consciousness/master': IconGrid,
  '/consciousness/training': IconTraining,
  '/consciousness/playground': IconSparkle,
  '/consciousness/analytics': IconChart,
  '/consciousness/statistics': IconChart,
  '/consciousness/history': IconClock,
  '/consciousness/personality': IconBrain,
  '/consciousness/settings': IconSettings,
  '/consciousness/all-settings': IconSettings,
  '/consciousness/testing': IconBeaker,
  '/consciousness/test-runner': IconBeaker,
  '/consciousness/benchmark': IconActivity,
  '/consciousness/docs': IconCode,
  '/consciousness/api-explorer': IconCode,
  '/consciousness/debug': IconCode,
  '/consciousness/monitor': IconActivity,
  '/consciousness/insights': IconSparkle,
  '/consciousness/help': IconBookmark,
  '/consciousness/compare': IconChart,
  '/consciousness/alerts': IconActivity,
  '/consciousness/export': IconCode,
  '/consciousness/health': IconActivity,
  '/consciousness/versions': IconClock,
  '/benchmark': IconBenchmark,
  '/monitoring': IconActivity,
  '/settings': IconSettings,
  '/developer': IconCode,
  '/feedback': IconThumbUp,
  '/planner': IconGrid,
  '/phoneme': IconTokenizer,
  '/workspace-dashboard': IconActivity,
  '/members': IconAgents,
  '/usage': IconChart,
  '/audit-trail': IconActivity,
  '/workspace-settings': IconSettings,
  '/api-keys': IconCode,
  '/notifications': IconActivity,
  '/workspace-search': IconSearch,
  '/permissions': IconSettings,
  '/shared-data': IconUsers,
  '/plugins-cloud': IconSettings,
  '/shortcuts': IconSettings,
  '/writing': IconDocument,
  '/translate': IconChat,
  '/rewrite': IconSparkle,
  '/brainstorm': IconBrain,
  '/decide': IconChart,
  '/explain': IconSearch,
  '/wellness': IconSparkle,
}
