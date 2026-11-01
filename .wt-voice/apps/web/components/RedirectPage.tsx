'use client'
import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export function RedirectPage({ to }: { to: string }) {
  const router = useRouter()
  useEffect(() => { router.replace(to) }, [router, to])
  return <PageSkeleton cards={2} />
}
