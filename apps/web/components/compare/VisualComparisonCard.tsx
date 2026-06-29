'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

interface VisualComparisonCardProps {
  chartData: { name: string; throughput: number; latency: number; memory: number }[]
}

export default function VisualComparisonCard({ chartData }: VisualComparisonCardProps) {
  if (chartData.length < 2) return null

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
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid var(--border)' }} formatter={(val: number) => [val.toFixed(1) + ' tok/s', 'Throughput']} />
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
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid var(--border)' }} formatter={(val: number) => [val.toFixed(0) + ' ms', 'Latency']} />
                <Bar dataKey="latency" fill="var(--warning)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
