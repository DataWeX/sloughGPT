'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { StatCard, KpiGrid, ProgressBar } from '@sloughgpt/strui'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import type { TrainingReport, TrainingStatus } from '@/lib/multimodal-controller'

interface TrainingCardProps {
  report: TrainingReport
  trainStatus?: TrainingStatus | null
}

export default function TrainingCard({ report, trainStatus }: TrainingCardProps) {
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Training</CardTitle></CardHeader>
      <CardContent>
        <KpiGrid columns={4}>
          <StatCard label="Images learned" value={report.images_learned} />
          <StatCard label="Vocabulary" value={`${report.vocab_size} words`} />
          <StatCard label="Unique captions" value={report.unique_captions} />
          <StatCard label="Diversity" value={`${(report.diversity_ratio * 100).toFixed(0)}%`} />
        </KpiGrid>
        {trainStatus && trainStatus.total > 0 && (
          <div className="mt-3 space-y-1">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{trainStatus.completed}/{trainStatus.total} images</span>
              <span>{trainStatus.progress_pct}%</span>
            </div>
            <ProgressBar value={trainStatus.progress_pct} max={100} variant="default" />
            {trainStatus.current_caption && (
              <p className="text-[10px] text-muted-foreground/60 truncate">Current: {trainStatus.current_caption}</p>
            )}
          </div>
        )}
        {report.accuracy_history.length > 1 && (
          <div className="mt-4">
            <p className="text-xs font-medium text-muted-foreground mb-2">Accuracy over time</p>
            <div className="h-32">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={report.accuracy_history.map((acc, i) => ({ step: i + 1, accuracy: acc }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={10} />
                  <YAxis stroke="var(--muted-foreground)" fontSize={10} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '6px', fontSize: '11px' }} labelStyle={{ color: 'var(--foreground)' }} />
                  <Line type="monotone" dataKey="accuracy" stroke="var(--primary)" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        {report.caption_history.length > 0 && (
          <div className="mt-3">
            <p className="text-xs font-medium text-muted-foreground mb-1">Recent captions</p>
            <div className="max-h-24 overflow-y-auto space-y-0.5">
              {report.caption_history.slice(-6).reverse().map((c, i) => (
                <p key={i} className="text-[10px] text-muted-foreground/70 leading-relaxed border-l-2 border-primary/20 pl-2">{c}</p>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
