'use client'

// Evolution charts: beliefs line chart, qualia area chart, growth trajectory.
// Extracted from page.tsx by the page-decomposition refactor. Series colors
// come from dashboard-data (theme chart tokens), never hardcoded hues.

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  SectionHeader,
} from '@sloughgpt/strui'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts'
import {
  BELIEF_COLORS,
  QUALIA_COLORS,
  type BeliefsChartPoint,
  type GrowthChartPoint,
  type QualiaChartPoint,
} from '../dashboard-data'

interface DashboardChartsProps {
  beliefsData: BeliefsChartPoint[]
  qualiaData: QualiaChartPoint[]
  growthData: GrowthChartPoint[]
}

export function DashboardCharts({ beliefsData, qualiaData, growthData }: DashboardChartsProps) {
  return (
    <>
      {/* Evolution */}
      <SectionHeader
        title="Evolution"
        description="Beliefs, qualia, and growth over interactions"
      />
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
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 10 }}
                  stroke="rgb(var(--muted-foreground))"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgb(var(--card))',
                    border: '1px solid rgb(var(--border))',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
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
              Need at least 2 data points. Chat with consciousness enabled to generate data.
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
                <YAxis
                  domain={[-1, 1]}
                  tick={{ fontSize: 10 }}
                  stroke="rgb(var(--muted-foreground))"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgb(var(--card))',
                    border: '1px solid rgb(var(--border))',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
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
              Need at least 2 data points. Chat with consciousness enabled to generate data.
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
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="rgb(var(--muted-foreground))"
                />
                <YAxis tick={{ fontSize: 10 }} stroke="rgb(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgb(var(--card))',
                    border: '1px solid rgb(var(--border))',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
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
