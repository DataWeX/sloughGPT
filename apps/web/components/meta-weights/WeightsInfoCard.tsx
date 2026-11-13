'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

export function WeightsInfoCard() {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">How It Works</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5">
        <div className="text-[10px] text-muted-foreground/60 space-y-1">
          <p>Meta-weights adjust inference parameters based on feedback from similar conversations. When you rate a response, the system learns which settings produce better outputs.</p>
          <ul className="list-disc list-inside space-y-0.5 text-[10px]">
            <li><strong>Temperature</strong> — controls randomness (higher = more creative)</li>
            <li><strong>Top P</strong> — nucleus sampling threshold</li>
            <li><strong>Repetition Penalty</strong> — discourages repeated phrases</li>
            <li><strong>Style Bias</strong> — shifts between formal and casual tone</li>
            <li><strong>Confidence Boost</strong> — increases certainty in responses</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}
