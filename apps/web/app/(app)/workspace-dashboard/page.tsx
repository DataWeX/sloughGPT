import { redirect } from '@/vite/next-compat/navigation'

export default function RedirectPage(): never {
  redirect('/workspace')
}
