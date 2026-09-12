'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { PageContainer } from '@/components/PageContainer'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'

interface FaqItem {
  qKey: string
  aKey: string
}

const FAQ_ITEMS: FaqItem[] = [
  { qKey: 'consciousness_help.faq_q1', aKey: 'consciousness_help.faq_a1' },
  { qKey: 'consciousness_help.faq_q2', aKey: 'consciousness_help.faq_a2' },
  { qKey: 'consciousness_help.faq_q3', aKey: 'consciousness_help.faq_a3' },
  { qKey: 'consciousness_help.faq_q4', aKey: 'consciousness_help.faq_a4' },
  { qKey: 'consciousness_help.faq_q5', aKey: 'consciousness_help.faq_a5' },
  { qKey: 'consciousness_help.faq_q6', aKey: 'consciousness_help.faq_a6' },
  { qKey: 'consciousness_help.faq_q7', aKey: 'consciousness_help.faq_a7' },
  { qKey: 'consciousness_help.faq_q8', aKey: 'consciousness_help.faq_a8' },
  { qKey: 'consciousness_help.faq_q9', aKey: 'consciousness_help.faq_a9' },
  { qKey: 'consciousness_help.faq_q10', aKey: 'consciousness_help.faq_a10' },
]

const QUICK_LINKS = [
  { path: '/consciousness/quickstart', labelKey: 'consciousness_help.link_quickstart', descKey: 'consciousness_help.link_quickstart_desc' },
  { path: '/consciousness/dashboard', labelKey: 'consciousness_help.link_dashboard', descKey: 'consciousness_help.link_dashboard_desc' },
  { path: '/consciousness/monitor', labelKey: 'consciousness_help.link_monitor', descKey: 'consciousness_help.link_monitor_desc' },
  { path: '/consciousness/insights', labelKey: 'consciousness_help.link_insights', descKey: 'consciousness_help.link_insights_desc' },
  { path: '/consciousness/personality', labelKey: 'consciousness_help.link_personality', descKey: 'consciousness_help.link_personality_desc' },
  { path: '/consciousness/training', labelKey: 'consciousness_help.link_training', descKey: 'consciousness_help.link_training_desc' },
  { path: '/consciousness/docs', labelKey: 'consciousness_help.link_api', descKey: 'consciousness_help.link_api_desc' },
  { path: '/consciousness/settings', labelKey: 'consciousness_help.link_settings', descKey: 'consciousness_help.link_settings_desc' },
  { path: '/consciousness/playground', labelKey: 'consciousness_help.link_playground', descKey: 'consciousness_help.link_playground_desc' },
  { path: '/consciousness/analytics', labelKey: 'consciousness_help.link_analytics', descKey: 'consciousness_help.link_analytics_desc' },
  { path: '/consciousness/history', labelKey: 'consciousness_help.link_history', descKey: 'consciousness_help.link_history_desc' },
  { path: '/consciousness/benchmark', labelKey: 'consciousness_help.link_benchmark', descKey: 'consciousness_help.link_benchmark_desc' },
]

const KEYBOARD_SHORTCUTS = [
  { shortcut: 'Ctrl+Shift+C', actionKey: 'consciousness_shortcuts.toggle' },
  { shortcut: 'Ctrl+Shift+R', actionKey: 'consciousness_shortcuts.reflect' },
  { shortcut: 'Ctrl+Shift+D', actionKey: 'consciousness_shortcuts.dashboard' },
  { shortcut: 'Ctrl+Shift+P', actionKey: 'consciousness_shortcuts.personality' },
  { shortcut: 'Ctrl+Shift+Q', actionKey: 'consciousness_shortcuts.quickactions' },
]

export default function ConsciousnessHelpPage() {
  const router = useRouter()
  const { t } = useLocale()
  const [openFaq, setOpenFaq] = useState<number | null>(null)

  return (
    <PageContainer title={t('consciousness_help.page_title')}>
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_help.faq_title')}</CardTitle>
            <CardDescription>{t('consciousness_help.faq_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {FAQ_ITEMS.map((item, idx) => (
              <div key={idx} className="rounded-md border border-border/50">
                <button
                  type="button"
                  onClick={() => setOpenFaq(openFaq === idx ? null : idx)}
                  className="w-full flex items-center justify-between px-4 py-3 text-left text-sm font-medium hover:bg-muted/30 transition-colors"
                >
                  <span>{t(item.qKey)}</span>
                  <span className="text-muted-foreground text-xs">{openFaq === idx ? '−' : '+'}</span>
                </button>
                {openFaq === idx && (
                  <div className="px-4 pb-3 text-xs text-muted-foreground leading-relaxed">
                    {t(item.aKey)}
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_help.quick_links_title')}</CardTitle>
            <CardDescription>{t('consciousness_help.quick_links_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              {QUICK_LINKS.map((link) => (
                <button
                  key={link.path}
                  onClick={() => router.push(link.path)}
                  className="rounded-md border p-3 text-left text-sm hover:bg-muted transition-colors"
                >
                  <p className="font-medium">{t(link.labelKey)}</p>
                  <p className="text-[10px] text-muted-foreground">{t(link.descKey)}</p>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_help.shortcuts_title')}</CardTitle>
            <CardDescription>{t('consciousness_help.shortcuts_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {KEYBOARD_SHORTCUTS.map((item) => (
                <div key={item.shortcut} className="flex items-center justify-between rounded-md border border-border/50 px-4 py-2">
                  <span className="text-sm">{t(item.actionKey)}</span>
                  <Badge variant="outline" className="font-mono text-xs">{item.shortcut}</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_help.system_info_title')}</CardTitle>
            <CardDescription>{t('consciousness_help.system_info_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between rounded-md border border-border/50 px-4 py-2">
                <span className="text-muted-foreground">{t('consciousness_help.version')}</span>
                <span className="font-mono text-xs">3.0.0</span>
              </div>
              <div className="flex items-center justify-between rounded-md border border-border/50 px-4 py-2">
                <span className="text-muted-foreground">{t('consciousness_help.api_endpoint')}</span>
                <span className="font-mono text-xs">/consciousness</span>
              </div>
              <div className="flex items-center justify-between rounded-md border border-border/50 px-4 py-2">
                <span className="text-muted-foreground">{t('consciousness_help.total_pages')}</span>
                <span className="font-mono text-xs">26</span>
              </div>
              <div className="flex items-center justify-between rounded-md border border-border/50 px-4 py-2">
                <span className="text-muted-foreground">{t('consciousness_help.total_endpoints')}</span>
                <span className="font-mono text-xs">37</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
