'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts'

interface VisualComparisonCardProps {
  chartData: { name: string; throughput: number; latency: number; memory: number }[]
}

export default function VisualComparisonCard({ chartData }: VisualComparisonCardProps) {
  if (chartData.length < 2) return null

  const maxThroughput = Math.max(...chartData.map(d => d.throughput))
  const maxLatency = Math.max(...chartData.map(d => d.latency))
  const maxMemory = Math.max(...chartData.map(d => d.memory))

  const radarData = chartData.map(d => ({
    name: d.name,
    throughput: maxThroughput > 0 ? (d.throughput / maxThroughput) * 100 : 0,
    latency: maxLatency > 0 ? ((maxLatency - d.latency) / maxLatency) * 100 : 0,
    memory: maxMemory > 0 ? ((maxMemory - d.memory) / maxMemory) * 100 : 0,
  }))

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Visual Comparison</CardTitle></CardHeader>
      <CardContent className="space-y-6">
        <div>
          <p className="text-xs text-muted-foreground mb-2">Throughput (tok/s) — higher is better</p>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid var(--border)' }} formatter={(val: any) => [Number(val).toFixed(1) + ' tok/s', 'Throughput']} />
                <Bar dataKey="throughput" fill="var(--primary)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-2">Average latency (ms) — lower is better</p>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" hide />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid var(--border)' }} formatter={(val: any) => [Number(val).toFixed(0) + ' ms', 'Latency']} />
                <Bar dataKey="latency" fill="var(--warning)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-2">Memory usage (MB) — lower is better</p>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" hide />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid var(--border)' }} formatter={(val: any) => [Number(val).toFixed(0) + ' MB', 'Memory']} />
                <Bar dataKey="memory" fill="var(--accent)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        {chartData.length >= 3 && (
          <div>
            <p className="text-xs text-muted-foreground mb-2">Overall profile (normalized, higher is better)</p>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="var(--border)" />
                  <PolarAngleAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <PolarRadiusAxis tick={{ fontSize: 9 }} domain={[0, 100]} />
                  <Radar name="Throughput" dataKey="throughput" stroke="var(--primary)" fill="var(--primary)" fillOpacity={0.15} />
                  <Radar name="Latency" dataKey="latency" stroke="var(--warning)" fill="var(--warning)" fillOpacity={0.15} />
                  <Radar name="Memory" dataKey="memory" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.15} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
