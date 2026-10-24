import { redirect } from 'next/navigation'

/** Model card generator now lives in the /training Results tab (ModelCardDialog). */
export default function ModelCardRedirectPage() {
  redirect('/training?tab=results')
}
