'use client'
export const dynamic = 'force-dynamic'

import { useSearchParams } from '@/vite/next-compat/navigation'
import { PageContainer } from '@/components/PageContainer'
import { ModelsPanel } from '@/components/models/ModelsPanel'

/**
 * Models — the v1 route-map target (card 20261101_154054).
 *
 * Merges the old /models redirect, the /model/[id] detail route and the
 * Developer Models tab into one journey surface: catalog + engine cards +
 * a ?id=-driven detail fold. Old paths /infer and /registry 307 here via
 * lib/redirects.ts.
 */
export default function ModelsPage() {
  const searchParams = useSearchParams()
  const selectedId = searchParams.get('id')

  return (
    <PageContainer title="Models" subtitle="Browse, load and inspect your models">
      <ModelsPanel selectedId={selectedId} />
    </PageContainer>
  )
}
