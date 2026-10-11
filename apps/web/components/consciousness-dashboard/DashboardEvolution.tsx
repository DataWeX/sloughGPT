/**
 * Evolution — beliefs, qualia and growth charts over interactions.
 *
 * The three charts share one tooltip style; each degrades to a hint when it
 * has fewer than two data points (the growth chart hides entirely instead).
 */

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  SectionHeader,
} from '@sloughgpt/strui'
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
  BELIEF_COLORS,
  QUALIA_COLORS,
  type BeliefsChartData,
  type GrowthPoint,
  type QualiaChartData,
} from './types'

/** Recharts tooltip surface, themed from the design tokens. */
const TOOLTIP_STYLE = {
  backgroundColor: 'hsl(var(--card))',
  border: '1px solid hsl(var(--border))',
  borderRadius: '8px',
  fontSize: '12px',
}

const EMPTY_CHART_HINT =
  'Need at least 2 data points. Chat with consciousness enabled to generate data.'

export interface DashboardEvolutionProps {
  beliefsData: BeliefsChartData[]
  qualiaData: QualiaChartData[]
  growthData: GrowthPoint[]
}

export function DashboardEvolution({
  beliefsData,
  qualiaData,
  growthData,
}: DashboardEvolutionProps) {
  return (
    <>
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
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
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
              {EMPTY_CHART_HINT}
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
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis
                  domain={[-1, 1]}
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
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
              {EMPTY_CHART_HINT}
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
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 10 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Area
                  type="monotone"
                  dataKey="cumulative"
                  stroke="var(--chart-1)"
                  fill="var(--chart-1)"
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
