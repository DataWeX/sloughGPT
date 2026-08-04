'use client'

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'

interface ChartPoint {
  time: string
  cpu: number
  mem: number
  tokens?: number
  latency?: number
}

export function SystemChart({ data }: { data: ChartPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
        <YAxis yAxisId="left" domain={[0, 100]} tick={{ fontSize: 10 }} width={30} />
        <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} width={40} />
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6 }} />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        <Line yAxisId="left" type="monotone" dataKey="cpu" stroke="rgb(var(--primary))" strokeWidth={1.5} dot={false} name="CPU %" />
        <Line yAxisId="left" type="monotone" dataKey="mem" stroke="rgb(var(--warning))" strokeWidth={1.5} dot={false} name="Memory %" />
        <Line yAxisId="right" type="monotone" dataKey="tokens" stroke="rgb(var(--success))" strokeWidth={1.5} dot={false} name="tok/s" />
        <Line yAxisId="right" type="monotone" dataKey="latency" stroke="rgb(var(--destructive))" strokeWidth={1.5} dot={false} name="Latency ms" />
      </LineChart>
    </ResponsiveContainer>
  )
}
