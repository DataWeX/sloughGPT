import { redirect } from '@/vite/next-compat/navigation'

export default function ErrorsRedirect() {
  redirect('/monitoring')
}
