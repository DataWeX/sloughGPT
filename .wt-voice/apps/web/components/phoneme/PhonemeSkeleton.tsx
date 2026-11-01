'use client'

import { Card, CardContent, Skeleton } from '@sloughgpt/strui'

interface PhonemeSkeletonProps {
  variant?: 'encode' | 'score' | 'compare' | 'practice' | 'batch' | 'detect' | 'quiz' | 'synthesize' | 'history' | 'flashcard'
  rows?: number
}

function InputSkeleton() {
  return (
    <div className="flex gap-3">
      <Skeleton className="h-10 flex-1 rounded" />
      <Skeleton className="h-10 w-[140px] rounded" />
      <Skeleton className="h-10 w-24 rounded" />
    </div>
  )
}

function PhonemeBadgesSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="flex flex-wrap gap-1">
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} className="h-6 w-12 rounded-full" />
      ))}
    </div>
  )
}

function ResultBlockSkeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-3 p-4 rounded-lg bg-muted/30">
      <Skeleton className="h-4 w-32" />
      <PhonemeBadgesSkeleton count={lines + 2} />
      <Skeleton className="h-4 w-48" />
      <Skeleton className="h-4 w-36" />
    </div>
  )
}

export default function PhonemeSkeleton({ variant = 'encode', rows = 3 }: PhonemeSkeletonProps) {
  if (variant === 'score') {
    return (
      <Card>
        <CardContent className="py-8 space-y-4">
          <div className="flex gap-3">
            <Skeleton className="h-10 flex-1 rounded" />
            <Skeleton className="h-10 flex-1 rounded" />
            <Skeleton className="h-10 w-[140px] rounded" />
            <Skeleton className="h-10 w-24 rounded" />
          </div>
          <div className="space-y-3 p-4 rounded-lg bg-muted/30">
            <div className="grid grid-cols-3 gap-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="text-center space-y-1">
                  <Skeleton className="h-8 w-16 mx-auto" />
                  <Skeleton className="h-3 w-12 mx-auto" />
                </div>
              ))}
            </div>
            <Skeleton className="h-2 w-full rounded-full" />
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Skeleton className="h-4 w-28" />
                <PhonemeBadgesSkeleton count={4} />
              </div>
              <div className="space-y-2">
                <Skeleton className="h-4 w-28" />
                <PhonemeBadgesSkeleton count={4} />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (variant === 'compare') {
    return (
      <Card>
        <CardContent className="py-8 space-y-4">
          <div className="flex gap-3">
            <Skeleton className="h-10 flex-1 rounded" />
            <Skeleton className="h-10 flex-1 rounded" />
            <Skeleton className="h-10 w-[140px] rounded" />
            <Skeleton className="h-10 w-24 rounded" />
          </div>
          <div className="space-y-4 p-4 rounded-lg bg-muted/30">
            <div className="grid grid-cols-2 gap-6">
              {Array.from({ length: 2 }).map((_, i) => (
                <div key={i} className="space-y-2">
                  <Skeleton className="h-4 w-20" />
                  <PhonemeBadgesSkeleton count={4} />
                </div>
              ))}
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-4 w-12" />
              </div>
              <Skeleton className="h-2 w-full rounded-full" />
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (variant === 'practice') {
    return (
      <div className="space-y-4">
        <Card>
          <CardContent className="py-8 space-y-4">
            <div className="flex gap-3">
              <Skeleton className="h-10 w-[140px] rounded" />
              <Skeleton className="h-10 w-32 rounded" />
            </div>
            <Skeleton className="h-10 w-full rounded" />
            <div className="p-3 rounded-lg bg-muted/30 space-y-2">
              <div className="flex items-center gap-2">
                <Skeleton className="h-4 w-12" />
                <Skeleton className="h-5 w-24" />
                <Skeleton className="h-5 w-10 rounded-full" />
              </div>
              <PhonemeBadgesSkeleton count={4} />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-8 space-y-4">
            <Skeleton className="h-10 w-full rounded" />
            <div className="flex gap-3">
              <Skeleton className="h-10 flex-1 rounded" />
              <Skeleton className="h-10 w-20 rounded" />
              <Skeleton className="h-10 w-28 rounded" />
            </div>
            <div className="p-3 rounded-lg bg-muted/30 space-y-2">
              <Skeleton className="h-4 w-24" />
              <PhonemeBadgesSkeleton count={3} />
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (variant === 'batch') {
    return (
      <Card>
        <CardContent className="py-8 space-y-4">
          <div className="flex gap-3">
            <Skeleton className="h-10 w-[140px] rounded" />
            <Skeleton className="h-10 w-[140px] rounded" />
          </div>
          <Skeleton className="h-24 w-full rounded" />
          <Skeleton className="h-10 w-28 rounded" />
          <div className="space-y-2">
            {Array.from({ length: rows }).map((_, i) => (
              <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-muted/30">
                <Skeleton className="h-4 w-20" />
                <PhonemeBadgesSkeleton count={3} />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (variant === 'quiz') {
    return (
      <div className="space-y-4">
        <Card>
          <CardContent className="py-8 space-y-3">
            <div className="flex items-center justify-between">
              <Skeleton className="h-6 w-32" />
              <div className="flex gap-2">
                <Skeleton className="h-5 w-16 rounded-full" />
                <Skeleton className="h-5 w-16 rounded-full" />
              </div>
            </div>
            <Skeleton className="h-10 w-40 rounded" />
            <PhonemeBadgesSkeleton count={4} />
            <div className="flex gap-2">
              <Skeleton className="h-10 w-28 rounded" />
              <Skeleton className="h-10 w-24 rounded" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-8 space-y-4">
            <div className="flex gap-3">
              <Skeleton className="h-10 w-[140px] rounded" />
              <Skeleton className="h-10 w-[140px] rounded" />
              <Skeleton className="h-10 w-28 rounded" />
            </div>
            <div className="p-3 rounded-lg bg-muted/30 space-y-2">
              <Skeleton className="h-4 w-28" />
              <PhonemeBadgesSkeleton count={4} />
            </div>
            <div className="flex gap-3">
              <Skeleton className="h-10 flex-1 rounded" />
              <Skeleton className="h-10 w-20 rounded" />
              <Skeleton className="h-10 w-24 rounded" />
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (variant === 'flashcard') {
    return (
      <Card>
        <CardContent className="py-8 space-y-4">
          <div className="flex gap-3">
            <Skeleton className="h-10 w-[140px] rounded" />
            <Skeleton className="h-10 w-[140px] rounded" />
            <Skeleton className="h-10 w-24 rounded" />
          </div>
          <div className="flex justify-center gap-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="w-3 h-3 rounded-full" />
            ))}
          </div>
          <Skeleton className="h-64 w-full rounded-xl" />
          <div className="flex justify-center gap-3">
            <Skeleton className="h-12 w-36 rounded" />
            <Skeleton className="h-12 w-28 rounded" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (variant === 'history') {
    return (
      <Card>
        <CardContent className="py-8 space-y-3">
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-40" />
            <Skeleton className="h-4 w-20" />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="text-center p-2 rounded-lg bg-background space-y-1">
                <Skeleton className="h-6 w-12 mx-auto" />
                <Skeleton className="h-3 w-16 mx-auto" />
              </div>
            ))}
          </div>
          <Skeleton className="h-2 w-full rounded-full" />
          <div className="flex gap-2">
            <Skeleton className="h-9 w-[140px] rounded" />
            <Skeleton className="h-9 w-20 rounded" />
            <Skeleton className="h-9 w-20 rounded" />
            <Skeleton className="h-9 w-20 rounded" />
          </div>
          <div className="space-y-2">
            {Array.from({ length: rows }).map((_, i) => (
              <div key={i} className="flex items-center gap-3 p-2 rounded-lg">
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-4 w-4" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-6 w-12 rounded-full ml-auto" />
                <Skeleton className="h-6 w-10 rounded-full" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (variant === 'synthesize') {
    return (
      <Card>
        <CardContent className="py-8 space-y-4">
          <div className="flex gap-3">
            <Skeleton className="h-10 flex-1 rounded" />
            <Skeleton className="h-10 w-32 rounded" />
          </div>
          <div className="space-y-3 p-4 rounded-lg bg-muted/30">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-10 w-full rounded" />
            <Skeleton className="h-32 w-full rounded" />
          </div>
        </CardContent>
      </Card>
    )
  }

  // Default: encode
  return (
    <Card>
      <CardContent className="py-8 space-y-4">
        <InputSkeleton />
        <ResultBlockSkeleton lines={rows} />
      </CardContent>
    </Card>
  )
}
