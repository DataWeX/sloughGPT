'use client'

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

interface ChartPoint {
  time: string
  cpu: number
  mem: number
}

export function SystemChart({ data }: { data: ChartPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
        <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} width={30} />
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6 }} />
        <Line type="monotone" dataKey="cpu" stroke="rgb(var(--primary))" strokeWidth={1.5} dot={false} name="CPU %" />
        <Line type="monotone" dataKey="mem" stroke="rgb(var(--warning))" strokeWidth={1.5} dot={false} name="Memory %" />
      </LineChart>
    </ResponsiveContainer>
  )
}
