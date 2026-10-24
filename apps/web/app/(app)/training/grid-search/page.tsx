import { redirect } from 'next/navigation'

/** Grid Search now lives in the /training Train tab (mode selector). */
export default function GridSearchRedirectPage() {
  redirect('/training?mode=grid')
}
