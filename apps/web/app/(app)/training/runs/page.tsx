import { redirect } from 'next/navigation'

/** Runs list now lives in the /training Results tab (RunsSection). */
export default function RunsRedirectPage() {
  redirect('/training?tab=results')
}
