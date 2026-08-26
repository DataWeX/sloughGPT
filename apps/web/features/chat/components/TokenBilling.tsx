'use client'

import { useState, useCallback, useEffect, useMemo, memo } from 'react'
import { Button, IconX, IconRefresh, IconCheck, IconDownload } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface TokenBalance {
  userId: string
  balance: number
  dailyLimit: number
  dailyUsed: number
  monthlyLimit: number
  monthlyUsed: number
  tier: 'free' | 'pro' | 'enterprise'
}

interface UsageRecord {
  id: string
  userId: string
  model: string
  inputTokens: number
  outputTokens: number
  totalTokens: number
  cost: number
  timestamp: number
  requestId: string
}

interface UsageSummary {
  totalRequests: number
  totalTokens: number
  totalCost: number
  byModel: Record<string, { requests: number; tokens: number; cost: number }>
  byDay: Record<string, { requests: number; tokens: number }>
}

interface TokenBillingProps {
  className?: string
}

const TIER_LIMITS = {
  free: { daily: 500, monthly: 10000, price: 0 },
  pro: { daily: 10000, monthly: 300000, price: 20 },
  enterprise: { daily: 100000, monthly: 3000000, price: 100 },
}

const MODEL_PRICING = {
  'gpt-4': { input: 0.03, output: 0.06 },
  'gpt-4-turbo': { input: 0.01, output: 0.03 },
  'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
  'claude-3-opus': { input: 0.015, output: 0.075 },
  'claude-3-sonnet': { input: 0.003, output: 0.015 },
  'default': { input: 0.001, output: 0.002 },
}

function formatTokens(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return n.toString()
}

function formatCost(n: number): string {
  return `$${n.toFixed(4)}`
}

export const TokenBilling = memo(function TokenBilling({
  className,
}: TokenBillingProps) {
  const [balance, setBalance] = useState<TokenBalance | null>(null)
  const [usage, setUsage] = useState<UsageSummary | null>(null)
  const [history, setHistory] = useState<UsageRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'overview' | 'history' | 'pricing'>('overview')

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [balanceRes, usageRes, historyRes] = await Promise.all([
        fetch('/api/tokens/balance').then(r => r.json()),
        fetch('/api/tokens/usage/summary').then(r => r.json()),
        fetch('/api/tokens/usage/history?limit=50').then(r => r.json()),
      ])
      setBalance(balanceRes)
      setUsage(usageRes)
      setHistory(historyRes.records || [])
    } catch (e) {
      console.error('Failed to fetch token data:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const dailyPercentage = useMemo(() => {
    if (!balance) return 0
    return Math.min(100, (balance.dailyUsed / balance.dailyLimit) * 100)
  }, [balance])

  const monthlyPercentage = useMemo(() => {
    if (!balance) return 0
    return Math.min(100, (balance.monthlyUsed / balance.monthlyLimit) * 100)
  }, [balance])

  if (loading) {
    return (
      <div className={cn('border rounded-lg bg-card p-4', className)}>
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-muted rounded w-1/3"></div>
          <div className="h-8 bg-muted rounded w-1/2"></div>
          <div className="h-4 bg-muted rounded w-2/3"></div>
        </div>
      </div>
    )
  }

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Token Billing</span>
          {balance && (
            <span className={cn(
              'text-[10px] px-2 py-0.5 rounded-full font-medium',
              balance.tier === 'free' && 'bg-muted text-muted-foreground',
              balance.tier === 'pro' && 'bg-primary/20 text-primary',
              balance.tier === 'enterprise' && 'bg-success/20 text-success',
            )}>
              {balance.tier.charAt(0).toUpperCase() + balance.tier.slice(1)}
            </span>
          )}
        </div>
        <Button variant="ghost" size="icon-sm" className="h-6 w-6" onClick={fetchData}>
          <IconRefresh className="h-3 w-3" />
        </Button>
      </div>

      <div className="flex border-b">
        {(['overview', 'history', 'pricing'] as const).map(tab => (
          <button
            key={tab}
            type="button"
            onClick={() => setActiveTab(tab)}
            className={cn(
              'flex-1 text-xs py-2 capitalize transition-colors',
              activeTab === tab
                ? 'border-b-2 border-primary text-primary'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="p-4">
        {activeTab === 'overview' && balance && (
          <div className="space-y-4">
            <div className="text-center">
              <div className="text-3xl font-bold">{formatTokens(balance.balance)}</div>
              <div className="text-xs text-muted-foreground">tokens remaining</div>
            </div>

            <div className="space-y-3">
              <UsageBar
                label="Daily Usage"
                used={balance.dailyUsed}
                limit={balance.dailyLimit}
                percentage={dailyPercentage}
              />
              <UsageBar
                label="Monthly Usage"
                used={balance.monthlyUsed}
                limit={balance.monthlyLimit}
                percentage={monthlyPercentage}
              />
            </div>

            {usage && (
              <div className="grid grid-cols-3 gap-2 pt-2">
                <StatCard label="Requests" value={usage.totalRequests.toString()} />
                <StatCard label="Tokens" value={formatTokens(usage.totalTokens)} />
                <StatCard label="Cost" value={formatCost(usage.totalCost)} />
              </div>
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {history.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-4">No usage yet</p>
            ) : (
              history.map(record => (
                <div key={record.id} className="flex items-center justify-between py-2 border-b last:border-0">
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium truncate">{record.model}</div>
                    <div className="text-[10px] text-muted-foreground">
                      {new Date(record.timestamp).toLocaleString()}
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-xs">{formatTokens(record.totalTokens)}</div>
                    <div className="text-[10px] text-muted-foreground">{formatCost(record.cost)}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'pricing' && (
          <div className="space-y-3">
            {Object.entries(TIER_LIMITS).map(([tier, limits]) => (
              <div
                key={tier}
                className={cn(
                  'border rounded-lg p-3',
                  balance?.tier === tier && 'border-primary bg-primary/5',
                )}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium capitalize">{tier}</span>
                  <span className="text-xs font-bold">
                    {limits.price === 0 ? 'Free' : `$${limits.price}/mo`}
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground space-y-1">
                  <div>Daily: {formatTokens(limits.daily)} tokens</div>
                  <div>Monthly: {formatTokens(limits.monthly)} tokens</div>
                </div>
                {balance?.tier === tier && (
                  <div className="mt-2 text-[10px] text-primary">Current plan</div>
                )}
              </div>
            ))}

            <div className="pt-2">
              <h4 className="text-xs font-medium mb-2">Model Pricing (per 1K tokens)</h4>
              <div className="space-y-1">
                {Object.entries(MODEL_PRICING).filter(([k]) => k !== 'default').map(([model, pricing]) => (
                  <div key={model} className="flex justify-between text-[10px]">
                    <span className="text-muted-foreground">{model}</span>
                    <span>In: {formatCost(pricing.input)} / Out: {formatCost(pricing.output)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
})

function UsageBar({ label, used, limit, percentage }: {
  label: string
  used: number
  limit: number
  percentage: number
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px]">
        <span className="text-muted-foreground">{label}</span>
        <span>{formatTokens(used)} / {formatTokens(limit)}</span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full transition-all',
            percentage > 90 ? 'bg-destructive' :
            percentage > 70 ? 'bg-warning' :
            'bg-primary',
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center p-2 rounded bg-muted/30">
      <div className="text-sm font-medium">{value}</div>
      <div className="text-[10px] text-muted-foreground">{label}</div>
    </div>
  )
}