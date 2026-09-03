'use client'

import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, cn } from '@sloughgpt/strui'
import { IconChat, IconModels } from '@/components/icons/NavIcons'

interface StatsGridProps {
  apiStatus: string
  modelCount: number | null
  currentSoul: { name: string; description: string; traits: string[] } | null
  modelStatus: { loaded: boolean; model: string | null }
  inferenceCount: number | null
  t: (key: string) => string
}

export function StatsGrid({ apiStatus, modelCount, currentSoul, modelStatus, inferenceCount, t }: StatsGridProps) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Card className="flex flex-col justify-between">
        <CardHeader className="pb-1 px-4 pt-4">
          <CardDescription className="text-xs font-medium text-muted-foreground">{t('home.stats.status')}</CardDescription>
        </CardHeader>
        <CardContent className="pb-4 px-4">
          {apiStatus === 'loading' ? (
            <div className="h-6 w-20 animate-pulse rounded bg-muted" />
          ) : (
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/40" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
              </span>
              <p className="text-sm font-semibold">Online</p>
            </div>
          )}
        </CardContent>
      </Card>
      <Card className="flex flex-col justify-between">
        <CardHeader className="pb-1 px-4 pt-4">
          <CardDescription className="text-xs font-medium text-muted-foreground">{t('home.stats.models')}</CardDescription>
        </CardHeader>
        <CardContent className="pb-4 px-4">
          {apiStatus === 'loading' ? (
            <div className="h-6 w-12 animate-pulse rounded bg-muted" />
          ) : (
            <p className="text-sm font-semibold tabular-nums">{modelCount !== null ? modelCount : '\u2014'}</p>
          )}
        </CardContent>
      </Card>
      <Card className="flex flex-col justify-between">
        <CardHeader className="pb-1 px-4 pt-4">
          <CardDescription className="text-xs font-medium text-muted-foreground">{t('home.stats.personality')}</CardDescription>
        </CardHeader>
        <CardContent className="pb-4 px-4">
          <p className="text-sm font-semibold truncate">{currentSoul?.name || '\u2014'}</p>
        </CardContent>
      </Card>
      <Card className="bg-gradient-to-br from-accent/5 to-transparent border-accent/20 flex flex-col justify-between">
        <CardHeader className="pb-1 px-4 pt-4">
          <CardDescription className="text-xs font-medium text-muted-foreground">Active</CardDescription>
        </CardHeader>
        <CardContent className="pb-4 px-4">
          <p className="text-sm font-semibold truncate">{modelStatus.loaded ? `${modelStatus.model} + ${currentSoul?.name || 'default'}` : 'Not loaded'}</p>
          {inferenceCount !== null && inferenceCount !== undefined && (
            <p className="text-xs text-muted-foreground mt-0.5 tabular-nums">{inferenceCount} conversations</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
