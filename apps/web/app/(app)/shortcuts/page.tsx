'use client'
export const dynamic = 'force-dynamic'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  Keyboard, RefreshCw, Plus, Search, Command, ArrowUp,
  MessageSquare, Settings, Zap, Database, Brain, GraduationCap
} from 'lucide-react'

interface Shortcut {
  keys: string[]
  label: string
  category: string
  icon: typeof RefreshCw
}

const shortcuts: Shortcut[] = [
  // Global
  { keys: ['Ctrl', 'K'], label: 'Command palette', category: 'Global', icon: Command },
  { keys: ['Ctrl', '?'], label: 'Show keyboard shortcuts', category: 'Global', icon: Keyboard },
  { keys: ['Ctrl', '\\'], label: 'Toggle sidebar', category: 'Global', icon: Settings },
  { keys: ['Escape'], label: 'Cancel / dismiss / close', category: 'Global', icon: Settings },

  // Navigation
  { keys: ['1-9'], label: 'Navigate to page (when not in input)', category: 'Navigation', icon: ArrowUp },
  { keys: ['Ctrl', 'F'], label: 'Focus search', category: 'Navigation', icon: Search },

  // Refresh
  { keys: ['R'], label: 'Refresh data (most pages)', category: 'Actions', icon: RefreshCw },
  { keys: ['N'], label: 'New item (datasets, knowledge, agents)', category: 'Actions', icon: Plus },

  // Training
  { keys: ['Ctrl', 'Enter'], label: 'Start training', category: 'Training', icon: GraduationCap },
  { keys: ['Ctrl', 'Shift', 'T'], label: 'Open test model dialog', category: 'Training', icon: Zap },

  // Chat
  { keys: ['Ctrl', 'N'], label: 'New chat', category: 'Chat', icon: MessageSquare },
  { keys: ['Ctrl', 'R'], label: 'Regenerate response', category: 'Chat', icon: RefreshCw },
  { keys: ['Ctrl', 'Shift', 'R'], label: 'Rename conversation', category: 'Chat', icon: MessageSquare },
  { keys: ['Ctrl', 'Shift', 'E'], label: 'Export as Markdown', category: 'Chat', icon: MessageSquare },
  { keys: ['Ctrl', 'Shift', 'D'], label: 'Duplicate conversation', category: 'Chat', icon: MessageSquare },
  { keys: ['Ctrl', 'Shift', 'B'], label: 'Toggle bookmarks', category: 'Chat', icon: MessageSquare },
  { keys: ['Ctrl', 'Shift', 'N'], label: 'Add note to last message', category: 'Chat', icon: MessageSquare },
  { keys: ['/'], label: 'Focus search (when not in input)', category: 'Chat', icon: Search },
  { keys: ['Ctrl', 'Y'], label: 'Approve tool call', category: 'Chat', icon: Zap },
  { keys: ['↑'], label: 'Edit last message', category: 'Chat', icon: MessageSquare },

  // Datasets
  { keys: ['Ctrl', 'F'], label: 'Focus dataset search', category: 'Datasets', icon: Database },
  { keys: ['Double-click'], label: 'Rename dataset', category: 'Datasets', icon: Database },

  // Knowledge
  { keys: ['N'], label: 'New knowledge item', category: 'Knowledge', icon: Brain },
]

const categories = [...new Set(shortcuts.map(s => s.category))]

export default function ShortcutsPage() {
  return (
    <PageContainer title="Keyboard Shortcuts">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Keyboard Shortcuts" />}
      />

      <div className="space-y-6">
        <p className="text-sm text-muted-foreground">
          Keyboard shortcuts available across the application. Shortcuts are disabled when focus is in input fields.
        </p>

        {categories.map(cat => (
          <Card key={cat}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">{cat}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                {shortcuts.filter(s => s.category === cat).map((s, i) => (
                  <div key={i} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50">
                    <div className="flex items-center gap-2">
                      <s.icon className="h-3.5 w-3.5 text-muted-foreground" />
                      <span className="text-sm">{s.label}</span>
                    </div>
                    <div className="flex items-center gap-0.5">
                      {s.keys.map((key, ki) => (
                        <span key={ki}>
                          <kbd className="px-2 py-1 text-xs font-mono bg-muted border rounded shadow-sm">
                            {key}
                          </kbd>
                          {ki < s.keys.length - 1 && (
                            <span className="text-muted-foreground text-xs mx-0.5">+</span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </PageContainer>
  )
}
