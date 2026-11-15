'use client'
export const dynamic = 'force-dynamic'

/**
 * Settings — composition root.
 *
 * State and IO live in `useSettingsWorkbench`; rendering lives in the
 * `settings-*` siblings next to this file. Only the page shell belongs
 * here: the container and the section order.
 */
import { PageContainer } from '@/components/PageContainer'
import { useSettingsWorkbench } from './useSettingsWorkbench'
import { SettingsAppearance } from './settings-appearance'
import { SettingsConnection } from './settings-connection'
import { SettingsInteraction } from './settings-interaction'
import { SettingsSystem } from './settings-system'
import { SettingsProcess } from './settings-process'
import { SettingsTraining } from './settings-training'
import { SettingsDangerZone } from './settings-danger-zone'

export default function SettingsPage() {
  const wb = useSettingsWorkbench()

  return (
    <PageContainer title="Settings">
      <SettingsAppearance wb={wb} />
      <SettingsConnection wb={wb} />
      <SettingsInteraction wb={wb} />
      <SettingsSystem wb={wb} />
      <SettingsProcess wb={wb} />
      <SettingsTraining wb={wb} />
      <SettingsDangerZone wb={wb} />
    </PageContainer>
  )
}
