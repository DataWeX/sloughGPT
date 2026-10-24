import { redirect } from 'next/navigation'

/** Adaptive insights now live in the /training Results tab (InsightsSection). */
export default function InsightsRedirectPage() {
  redirect('/training?tab=results')
}
