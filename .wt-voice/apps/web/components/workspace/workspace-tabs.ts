import type { WorkspaceTab } from '@/components/workspace/WorkspaceSectionTabs'

export const overviewTabs: WorkspaceTab[] = [
  { path: '/workspace', labelKey: 'nav.workspace_dashboard' },
  { path: '/workspace/usage', labelKey: 'nav.usage' },
  { path: '/workspace/audit', labelKey: 'nav.audit_trail' },
]

export const membersTabs: WorkspaceTab[] = [
  { path: '/workspace/members', labelKey: 'nav.members' },
  { path: '/workspace/members/permissions', labelKey: 'nav.permissions' },
]

export const settingsTabs: WorkspaceTab[] = [
  { path: '/workspace/settings', labelKey: 'nav.workspace_settings' },
  { path: '/workspace/settings/api-keys', labelKey: 'nav.api_keys' },
  { path: '/workspace/settings/notifications', labelKey: 'nav.notifications' },
]

export const dataTabs: WorkspaceTab[] = [
  { path: '/workspace/data', labelKey: 'nav.shared_data' },
  { path: '/workspace/data/search', labelKey: 'nav.workspace_search' },
]
