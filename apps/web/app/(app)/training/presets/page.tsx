import { redirect } from 'next/navigation'

/** Presets now live in the /training Train tab (mode selector). */
export default function PresetsRedirectPage() {
  redirect('/training?mode=preset')
}
