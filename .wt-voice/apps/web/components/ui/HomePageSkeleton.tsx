'use client'

import { Skeleton } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader } from '@sloughgpt/strui'

export function ActiveModelBannerSkeleton() {
  return (
    <Card className="border-primary/20 bg-gradient-to-br from-primary/[0.04] via-transparent to-accent/[0.03]">
      <CardContent className="p-4 sm:p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1 space-y-2">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-3 w-28" />
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Skeleton className="h-7 w-16 rounded-md" />
            <Skeleton className="h-7 w-14 rounded-md" />
          </div>
        </div>
        <div className="flex items-center gap-4 mt-3 pt-3 border-t border-border/40">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-3 w-20" />
        </div>
      </CardContent>
    </Card>
  )
}

export function StatsGridSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <Card key={i} className="flex flex-col justify-between">
          <CardHeader className="pb-1 px-4 pt-4">
            <Skeleton className="h-3 w-16" />
          </CardHeader>
          <CardContent className="pb-4 px-4">
            <Skeleton className="h-5 w-14" />
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

export function FeedbackBarSkeleton() {
  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-3 flex-wrap">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-16" />
        </div>
      </CardContent>
    </Card>
  )
}

export function TrainingStatusSkeleton() {
  return (
    <Card className="border-primary/30 bg-primary/5">
      <CardContent className="py-3">
        <div className="flex items-center gap-3">
          <Skeleton className="h-2 w-2 rounded-full shrink-0" />
          <div className="min-w-0 flex-1 space-y-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-48" />
          </div>
          <Skeleton className="h-3 w-10 shrink-0" />
        </div>
      </CardContent>
    </Card>
  )
}

export function QuickActionsSkeleton() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1 space-y-1">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-3 w-40" />
            </div>
            <Skeleton className="h-8 w-20 rounded-md" />
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center gap-2 mb-2">
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-3 w-32" />
          </div>
          <div className="flex gap-2">
            <Skeleton className="h-9 flex-1 rounded-md" />
            <Skeleton className="h-8 w-14 rounded-md" />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export function RecentActivitySkeleton() {
  return (
    <Card>
      <CardContent className="py-3">
        <Skeleton className="h-4 w-28 mb-2" />
        <div className="space-y-1.5">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="flex items-center gap-2 px-1.5 py-1">
              <Skeleton className="h-1.5 w-1.5 rounded-full shrink-0" />
              <Skeleton className="h-3 flex-1" />
              <Skeleton className="h-3 w-16" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function UsageStatsSkeleton() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center gap-2 mb-2">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-20" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="space-y-1">
                <Skeleton className="h-5 w-12" />
                <Skeleton className="h-3 w-16" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center gap-2 mb-2">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-14" />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="space-y-1">
                <Skeleton className="h-5 w-12" />
                <Skeleton className="h-3 w-16" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export function SystemHealthSkeleton() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center gap-2 mb-2">
            <Skeleton className="h-3 w-12" />
            <Skeleton className="h-3 w-14" />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="space-y-1">
                <Skeleton className="h-4 w-10" />
                <Skeleton className="h-3 w-12" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      <div className="rounded-lg border border-border/60 p-3 sm:p-4 space-y-2">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-3/4" />
      </div>
    </div>
  )
}

export function NavigationGridSkeleton() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 sm:gap-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4">
          <div className="flex items-center gap-2 sm:gap-3">
            <Skeleton className="h-8 w-8 sm:h-9 sm:w-9 rounded-lg shrink-0" />
            <div className="min-w-0 space-y-1">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-3 w-28 hidden sm:block" />
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
