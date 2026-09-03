'use client'

import Link from 'next/link'
import { IconChat, IconModels } from '@/components/icons/NavIcons'
import { IconChevronRight, IconSearch, IconBolt, IconChart } from '@sloughgpt/strui'
import { formatBytes } from '@/lib/format-bytes'

interface NavigationGridProps {
  apiStatus: string
  modelStatus: { loaded: boolean; model: string | null }
  datasetStats: { totalDatasets: number; totalSize: number; totalSamples: number } | null
}

export function NavigationGrid({ apiStatus, modelStatus, datasetStats }: NavigationGridProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 sm:gap-3">
      <Link
        href="/chat"
        className="group relative overflow-hidden rounded-lg border border-accent/25 bg-gradient-to-br from-accent/8 via-accent/3 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-accent/8 hover:border-accent/40"
      >
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-accent/15 text-accent">
            <IconChat className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
          <div className="min-w-0">
            <p className="text-xs sm:text-sm font-semibold">Start chatting</p>
            <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Ask anything, get answers</p>
          </div>
        </div>
        <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-accent/50 transition-colors">
          <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
        </div>
      </Link>
      <Link
        href="/models"
        className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
      >
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <IconModels className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
          <div className="min-w-0">
            <p className="text-xs sm:text-sm font-semibold">Personalities</p>
            <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Switch your agent&apos;s personality</p>
          </div>
        </div>
        <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
          <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
        </div>
      </Link>
      <Link
        href="/datasets"
        className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
      >
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-muted-foreground/10 text-muted-foreground">
            <IconSearch className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="text-xs sm:text-sm font-semibold">Datasets</p>
              {datasetStats && datasetStats.totalDatasets > 0 && (
                <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{datasetStats.totalDatasets}</span>
              )}
            </div>
            <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Manage training data</p>
          </div>
        </div>
        <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
          <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
        </div>
      </Link>
      <Link
        href="/training"
        className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
      >
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-accent/10 text-accent">
            <IconBolt className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="text-xs sm:text-sm font-semibold">Teach me</p>
              {datasetStats && datasetStats.totalDatasets > 0 && (
                <span className="text-xs px-1.5 py-0.5 rounded bg-accent/15 text-accent font-medium">{datasetStats.totalDatasets}</span>
              )}
            </div>
            <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Train from your writing</p>
          </div>
        </div>
        <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
          <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
        </div>
      </Link>
      <Link
        href="/monitoring"
        className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
      >
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-success/10 text-success">
            <IconChart className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
          <div className="min-w-0">
            <p className="text-xs sm:text-sm font-semibold">System Health</p>
            <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Monitor API and resources</p>
          </div>
        </div>
        <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
          <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
        </div>
      </Link>
      <Link
        href="/knowledge"
        className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
      >
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-primary/8 text-primary">
            <IconSearch className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
          <div className="min-w-0">
            <p className="text-xs sm:text-sm font-semibold">Knowledge</p>
            <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Facts the AI remembers</p>
          </div>
        </div>
        <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
          <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
        </div>
      </Link>
    </div>
  )
}
