import { redirect } from 'next/navigation'

/** Trends now live in the /training Results tab (InsightsSection). */
export default function TrendsRedirectPage() {
  redirect('/training?tab=results')
}
