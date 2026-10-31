'use client'

/**
 * The Evolution section: beliefs, qualia and growth charts.
 *
 * Chart series are derived here from the raw history payloads — timestamps get
 * their display labels and episodes get their cumulative growth — so the page
 * never has to know how a chart wants its data shaped.
 */
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  SectionHeader,
} from '@sloughgpt/strui'
import type { Episode, QualiaPoint, BeliefsPoint } from '@/hooks/useConsciousnessDashboard'
import { BELIEF_COLORS, QUALIA_COLORS, formatTime } from './dashboard-format'

interface DashboardEvolutionProps {
  qualiaHistory: QualiaPoint[]
  beliefsHistory: BeliefsPoint[]
  episodes: Episode[]
}

const CHART_TOOLTIP_STYLE = {
  backgroundColor: 'rgb(var(--card))',
  border: '1px solid rgb(var(--border))',
  borderRadius: '8px',
  fontSize: '12px',
}

const EMPTY_SERIES_MESSAGE =
  'Need at least 2 data points. Chat with consciousness enabled to generate data.'

export function DashboardEvolution({ qualiaHistory, beliefsHistory, episodes }: DashboardEvolutionProps) {
  // Qualia chart data with time labels
  const qualiaData = qualiaHistory.map((q) => ({
    ...q,
    time: formatTime(q.timestamp),
  }))

  // Beliefs chart data with time labels
  const beliefsData = beliefsHistory.map((b) => ({
    ...b,
    time: formatTime(b.timestamp),
  }))

  // Growth data from episodes
  const growthData = episodes.map((ep) => ({
    time: formatTime(ep.timestamp),
    growth: ep.growth_delta,
    cumulative: episodes
      .slice(0, episodes.indexOf(ep) + 1)
      .reduce((sum, e) => sum + e.growth_delta, 0),
  }))

  return (
    <>
      <SectionHeader title="Evolution" description="Beliefs, qualia, and growth over interactions" />

      <Card>
        <CardHeader>
          <CardTitle>Beliefs Evolution</CardTitle>
          <CardDescription>How self-beliefs change over interactions</CardDescription>
        </CardHeader>
        <CardContent>
          {beliefsData.length > 1 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={beliefsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="rgb(var(--muted-foreground))"
                />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} stroke="rgb(var(--muted-foreground))" />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                {Object.entries(BELIEF_COLORS).map(([key, color]) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stroke={color}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-48 sm:h-64 items-center justify-center text-sm text-muted-foreground">
              {EMPTY_SERIES_MESSAGE}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Qualia History</CardTitle>
          <CardDescription>Emotional and cognitive state over time</CardDescription>
        </CardHeader>
        <CardContent>
          {qualiaData.length > 1 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={qualiaData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="rgb(var(--muted-foreground))"
                />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 10 }} stroke="rgb(var(--muted-foreground))" />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                {Object.entries(QUALIA_COLORS).map(([key, color]) => (
                  <Area
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stroke={color}
                    fill={color}
                    fillOpacity={0.1}
                    strokeWidth={2}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-48 sm:h-64 items-center justify-center text-sm text-muted-foreground">
              {EMPTY_SERIES_MESSAGE}
            </div>
          )}
        </CardContent>
      </Card>

      {growthData.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle>Growth Trajectory</CardTitle>
            <CardDescription>Cumulative growth across interactions</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={growthData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="rgb(var(--muted-foreground))" />
                <YAxis tick={{ fontSize: 10 }} stroke="rgb(var(--muted-foreground))" />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Area
                  type="monotone"
                  dataKey="cumulative"
                  stroke="rgb(var(--chart-1))"
                  fill="rgb(var(--chart-1))"
                  fillOpacity={0.15}
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </>
  )
}
