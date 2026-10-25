import '../globals.css'
import AppLayout from '@/components/AppLayout'

export const dynamic = 'force-dynamic'

export default function AppLayoutWrapper({ children }: { children: React.ReactNode }) {
  // Consciousness quick actions live in the overlay rail now (AppLayout's
  // OverlayDock tenant) — no separate bottom-right FAB anymore.
  return <AppLayout>{children}</AppLayout>
}
