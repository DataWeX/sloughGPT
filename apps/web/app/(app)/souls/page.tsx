'use client'
export const dynamic = 'force-dynamic'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

/**
 * Consolidated: souls were renamed to personalities and merged into
 * /personality. Keep the route alive as a redirect so old links don't 404.
 */
export default function SoulsRedirect() {
  const router = useRouter()
  useEffect(() => {
    router.replace('/personality')
  }, [router])
  return null
}
