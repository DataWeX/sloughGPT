'use client'

import Link from 'next/link'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'
import { PUBLIC_API_URL } from '@/lib/config'

interface TrainingStatusProps {
  apiStatus: string
  modelStatus: { loaded: boolean; model: string | null }
  modelReadiness: { ready: boolean; phase: string; step: number; total: number; message: string }
  runningTraining: { name: string; status_message: string } | null
}

export function TrainingStatus({ apiStatus, modelStatus, modelReadiness, runningTraining }: TrainingStatusProps) {
  const { t } = useLocale()

  return (
    <>
      {apiStatus === 'offline' ? (
        modelReadiness.phase !== 'initializing' && modelReadiness.phase !== 'unknown' ? (
          <Card className="border-warning/35 bg-warning/5">
            <CardHeader>
              <CardTitle className="text-base">Starting up… ({modelReadiness.step}/{modelReadiness.total})</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <p>{modelReadiness.message}</p>
              <div className="h-1.5 w-full rounded-full bg-muted/60 overflow-hidden">
                <div className="h-full bg-warning rounded-full transition-all duration-500" style={{width: `${(modelReadiness.step / modelReadiness.total) * 100}%`}} />
              </div>
              <p className="text-[10px] text-muted-foreground/60">First startup may take 90 seconds while AI components load.</p>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-warning/35 bg-warning/5">
            <CardHeader>
              <CardTitle className="text-base">{t('home.apiOffline.title')}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm">
              {t('home.apiOffline.body', { url: PUBLIC_API_URL })}
            </CardContent>
          </Card>
        )
      ) : null}

      {apiStatus !== 'offline' && runningTraining && (
        <Link href="/training" className="block">
          <Card className="border-primary/30 bg-primary/5 cursor-pointer hover:border-primary/50 transition-colors">
            <CardContent className="py-3">
              <div className="flex items-center gap-3">
                <span className="relative flex h-2 w-2 shrink-0">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium truncate">Training: {runningTraining.name}</p>
                  <p className="text-[10px] text-muted-foreground truncate">{runningTraining.status_message}</p>
                </div>
                <span className="text-[10px] text-primary/60 shrink-0">View →</span>
              </div>
            </CardContent>
          </Card>
        </Link>
      )}

      {apiStatus === 'online' && !modelReadiness.ready && modelReadiness.phase !== 'initializing' && modelReadiness.phase !== 'unknown' && (
        <Card className="border-warning/30 bg-warning/5">
          <CardContent className="py-3">
            <div className="flex items-center gap-3">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-warning/60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-warning" />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">Model loading</p>
                <p className="text-[10px] text-muted-foreground">{modelReadiness.message}</p>
              </div>
              <div className="h-1.5 w-16 rounded-full bg-muted/60 overflow-hidden shrink-0">
                <div className="h-full bg-warning rounded-full transition-all duration-500" style={{width: `${(modelReadiness.step / modelReadiness.total) * 100}%`}} />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {apiStatus === 'online' && !modelStatus.loaded && (
        <Card className="border-dashed border-border/60 bg-muted/20">
          <CardContent className="py-3 flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">No model loaded</p>
              <p className="text-[10px] text-muted-foreground">Load a model in Models to start chatting</p>
            </div>
            <Link href="/models" className="inline-flex items-center h-7 px-2.5 rounded-md text-[11px] font-medium border border-border bg-background hover:bg-muted transition-colors shrink-0">
              Open Models
            </Link>
          </CardContent>
        </Card>
      )}
    </>
  )
}
