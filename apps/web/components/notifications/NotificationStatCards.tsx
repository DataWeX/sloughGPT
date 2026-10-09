import { Card, CardContent, KpiGrid, StatCard } from '@sloughgpt/strui'

export interface NotificationStatCardsProps {
  total: number
  training: number
  members: number
}

export function NotificationStatCards({ total, training, members }: NotificationStatCardsProps) {
  return (
    <KpiGrid className="mb-6">
      <StatCard label="Total" value={total} />
      <StatCard label="Training" value={training} />
      <StatCard label="Members" value={members} />
    </KpiGrid>
  )
}
