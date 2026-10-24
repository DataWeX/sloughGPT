import { redirect } from 'next/navigation'

/** Analytics now live in the /training Results tab (InsightsSection). */
export default function AnalyticsRedirectPage() {
  redirect('/training?tab=results')
}
