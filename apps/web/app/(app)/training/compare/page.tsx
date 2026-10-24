import { redirect } from 'next/navigation'

/** Run comparison now lives in the /training Results tab (CompareSection). */
export default function CompareRedirectPage() {
  redirect('/training?tab=results')
}
