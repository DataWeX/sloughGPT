'use client'

import { Skeleton } from '@sloughgpt/strui'
import { Card, CardContent } from '@sloughgpt/strui'

export function KnowledgeStatsSkeleton() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {Array.from({ length: 4 }).map((_, i) => (
        <Card key={i}>
          <CardContent className="p-4 space-y-2">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-6 w-12" />
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

export function KnowledgeCategoryChartSkeleton() {
  return (
    <Card>
      <CardContent className="p-4">
        <Skeleton className="h-4 w-32 mb-3" />
        <div className="flex items-end gap-1 h-24">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="flex-1 rounded-t" />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function KnowledgeTopicsSkeleton() {
  return (
    <Card>
      <CardContent className="p-3">
        <Skeleton className="h-3 w-28 mb-3" />
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex items-center gap-2">
              <Skeleton className="h-3 w-16" />
              <div className="flex-1 h-3 bg-muted/50 rounded-full overflow-hidden">
                <Skeleton className="h-full rounded-full w-3/4" />
              </div>
              <Skeleton className="h-3 w-6" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function KnowledgeAdapterSkeleton() {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-3 w-28" />
            <Skeleton className="h-5 w-14 rounded-full" />
          </div>
          <Skeleton className="h-7 w-20 rounded-md" />
        </div>
        <div className="flex gap-3 mt-2">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="h-3 w-32" />
        </div>
      </CardContent>
    </Card>
  )
}

export function KnowledgeRAGSkeleton() {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-5 w-16 rounded-full" />
          </div>
          <div className="flex gap-2">
            <Skeleton className="h-7 w-14 rounded-md" />
            <Skeleton className="h-7 w-16 rounded-md" />
            <Skeleton className="h-7 w-12 rounded-md" />
          </div>
        </div>
        <div className="flex gap-3 mt-2">
          <Skeleton className="h-3 w-20" />
          <Skeleton className="h-3 w-28" />
        </div>
      </CardContent>
    </Card>
  )
}

export function KnowledgeItemsSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="p-4 rounded-lg border border-border/60 bg-card space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <div className="flex gap-2">
            <Skeleton className="h-4 w-16 rounded" />
            <Skeleton className="h-4 w-12 rounded" />
            <Skeleton className="h-3 w-20" />
          </div>
        </div>
      ))}
    </div>
  )
}

export function KnowledgePageSkeleton() {
  return (
    <div className="space-y-4">
      <KnowledgeStatsSkeleton />
      <KnowledgeCategoryChartSkeleton />
      <KnowledgeAdapterSkeleton />
      <KnowledgeRAGSkeleton />
      <KnowledgeTopicsSkeleton />
      <KnowledgeItemsSkeleton />
    </div>
  )
}

export function DatasetListSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid gap-2">
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i}>
          <CardContent className="flex items-center justify-between py-3 px-4">
            <div className="min-w-0 flex-1 space-y-1.5">
              <Skeleton className="h-4 w-40" />
              <div className="flex gap-2">
                <Skeleton className="h-4 w-14 rounded" />
                <Skeleton className="h-4 w-10 rounded" />
              </div>
              <div className="flex gap-3">
                <Skeleton className="h-3 w-16" />
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-24" />
              </div>
            </div>
            <div className="flex items-center gap-1 ml-2 shrink-0">
              {Array.from({ length: 5 }).map((_, j) => (
                <Skeleton key={j} className="h-8 w-8 rounded" />
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

export function DatasetsPageSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Skeleton className="h-9 flex-1 max-w-xs rounded-md" />
        <Skeleton className="h-8 w-16 rounded-md" />
        <Skeleton className="h-8 w-16 rounded-md" />
        <Skeleton className="h-8 w-16 rounded-md" />
      </div>
      <DatasetListSkeleton />
    </div>
  )
}

export function ModelStatusCardSkeleton() {
  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center gap-3">
          <Skeleton className="h-10 w-10 rounded-lg shrink-0" />
          <div className="min-w-0 flex-1 space-y-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-48" />
          </div>
          <Skeleton className="h-8 w-20 rounded-md" />
        </div>
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="space-y-1">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-5 w-12" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function PersonalitiesCardSkeleton() {
  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <Skeleton className="h-4 w-28" />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="p-2.5 rounded-lg border border-border/40 space-y-1.5">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-3/4" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function ModelCatalogSkeleton() {
  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-3 w-16" />
        </div>
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3 py-2">
              <Skeleton className="h-8 w-8 rounded shrink-0" />
              <div className="flex-1 min-w-0 space-y-1">
                <Skeleton className="h-3.5 w-36" />
                <Skeleton className="h-3 w-24" />
              </div>
              <Skeleton className="h-6 w-14 rounded-full shrink-0" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function ModelsPageSkeleton() {
  return (
    <div className="space-y-4">
      <ModelStatusCardSkeleton />
      <PersonalitiesCardSkeleton />
      <ModelCatalogSkeleton />
    </div>
  )
}
