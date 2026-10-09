'use client'
export const dynamic = 'force-dynamic'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

/**
 * Consolidated: model catalog lives in Developer (Models tab),
 * personalities live in /personality. Keep the route alive as a redirect
 * so old links and shortcuts don't 404.
 */
export default function ModelsRedirect() {
  const router = useRouter()
  useEffect(() => {
    router.replace('/developer')
  }, [router])
  return null
}
