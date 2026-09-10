'use client'

import { KpiGrid, StatCard, StatusDot } from '@sloughgpt/strui'
import { IconModels, IconBrain, IconActivity, IconUsers } from '@sloughgpt/strui'

interface StatsGridProps {
  apiStatus: string
  modelCount: number | null
  currentSoul: { name: string; description: string; traits: string[] } | null
  modelStatus: { loaded: boolean; model: string | null }
  inferenceCount: number | null
  t: (key: string) => string
}

export function StatsGrid({ apiStatus, modelCount, currentSoul, modelStatus, inferenceCount, t }: StatsGridProps) {
  const loading = apiStatus === 'loading'
  
  return (
    <KpiGrid columns={4}>
      <StatCard
        label={t('home.stats.status')}
        value={loading ? '—' : 'Online'}
        icon={
          <StatusDot 
            tone={loading ? 'muted' : 'success'} 
            pulse={!loading} 
          />
        }
        loading={loading}
        className="transition-all duration-200 hover:shadow-md hover:shadow-success/10"
      />
      <StatCard
        label={t('home.stats.models')}
        value={loading ? '—' : (modelCount ?? '—')}
        icon={<IconModels className="h-4 w-4" />}
        loading={loading}
        numeric={!loading && modelCount !== null}
        className="transition-all duration-200 hover:shadow-md hover:shadow-primary/10"
      />
      <StatCard
        label={t('home.stats.personality')}
        value={loading ? '—' : (currentSoul?.name ?? '—')}
        icon={<IconBrain className="h-4 w-4" />}
        loading={loading}
        description={currentSoul?.traits?.[0]}
        className="transition-all duration-200 hover:shadow-md hover:shadow-accent/10"
      />
      <StatCard
        label="Active"
        value={loading ? '—' : (modelStatus.loaded ? 'Loaded' : 'Not loaded')}
        icon={<IconActivity className="h-4 w-4" />}
        loading={loading}
        description={
          inferenceCount !== null && inferenceCount !== undefined
            ? `${inferenceCount} conversations`
            : undefined
        }
        className="transition-all duration-200 hover:shadow-md hover:shadow-accent/10"
      />
    </KpiGrid>
  )
}
