import type { ReactNode } from 'react'

interface ChatLayoutProps {
  children: ReactNode
}

export default function ChatLayout({ children }: ChatLayoutProps) {
  return (
    <div className="flex flex-1 h-full min-h-0 w-full overflow-hidden">
      {children}
    </div>
  )
}
