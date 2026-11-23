'use client'

import { usePathname } from '@/vite/next-compat/navigation'
import { ConsciousnessQuickActions } from './ConsciousnessQuickActions'

/**
 * Route gate for the consciousness FAB.
 *
 * The FAB is a fixed bottom-right control (`bottom-20 right-6 z-[9999]`,
 * `lg:bottom-6` — bottom offsets are nav-aware: the 3.5rem bottom nav is
 * visible below lg, so every fixed bottom overlay clears it).
 * On `/chat` that corner belongs to the chat toast viewport
 * (`features/chat/components/feedback/Toast.tsx`: `bottom-20 right-0
 * lg:bottom-0 z-[300]`) and the composer — the FAB rendered on top of incoming toasts
 * ("Message deleted" was covered). Chat already exposes consciousness via
 * its toolbar toggle, so the global FAB simply doesn't mount there.
 */
export function ConsciousnessQuickActionsWrapper() {
  const pathname = usePathname()

  if (pathname === '/chat' || pathname.startsWith('/chat/')) return null

  return <ConsciousnessQuickActions />
}
