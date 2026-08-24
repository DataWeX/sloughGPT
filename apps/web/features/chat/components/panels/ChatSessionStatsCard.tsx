'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { chatDB } from '@/lib/db'
import { timeAgo } from '@/lib/time-ago'

interface ChatSessionStatsCardProps {
  sessionId: string | null
}

export function ChatSessionStatsCard({ sessionId }: ChatSessionStatsCardProps) {
  const [stats, setStats] = useState<{ totalSessions: number; totalMessages: number; avgMessages: number; lastActive: number | null } | null>(null)

  useEffect(() => {
    let ignore = false
    async function load() {
      try {
        const sessions = await chatDB.loadSessions()
        if (ignore) return
        const totalMessages = sessions.reduce((s, c) => s + (c.messages?.length ?? 0), 0)
        const avgMessages = sessions.length > 0 ? totalMessages / sessions.length : 0
        const lastActive = sessions.length > 0
          ? Math.max(...sessions.map(c => {
              const u = c.updatedAt ? new Date(c.updatedAt).getTime() : 0
              const cr = c.createdAt ? new Date(c.createdAt).getTime() : 0
              return Math.max(u, cr)
            }))
          : null
        setStats({ totalSessions: sessions.length, totalMessages, avgMessages, lastActive })
      } catch {
        if (!ignore) setStats(null)
      }
    }
    load()
    return () => { ignore = true }
  }, [sessionId])

  if (!stats || stats.totalSessions === 0) return null

  return (
    <Card data-testid="chat-session-stats">
      <CardHeader>
        <CardTitle className="text-base">Session Stats</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Sessions</div>
            <div className="text-lg font-semibold">{stats.totalSessions}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Messages</div>
            <div className="text-lg font-semibold">{stats.totalMessages}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Avg/Session</div>
            <div className="text-lg font-semibold">{stats.avgMessages.toFixed(1)}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Last Active</div>
            <div className="text-lg font-semibold">
              {stats.lastActive ? timeAgo(stats.lastActive) : '—'}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
