'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface TokenizerStats {
  vocab_size: number
  base_chars: number
  total_merges: number
  special_tokens: number
}

interface SampleWord {
  word: string
  tokens: string[]
  ids: number[]
}

interface TokenizerEfficiencyCardProps {
  stats: TokenizerStats | null
  samples: SampleWord[]
}

function computeEfficiency(stats: TokenizerStats | null, samples: SampleWord[]) {
  if (!stats) return null

  const vocabUtilization = stats.base_chars > 0
    ? (stats.base_chars / stats.vocab_size) * 100
    : 0

  const compressionRatio = samples.length > 0
    ? samples.reduce((s, w) => s + w.word.length, 0) /
      samples.reduce((s, w) => s + w.tokens.length, 0)
    : 0

  const avgTokensPerWord = samples.length > 0
    ? samples.reduce((s, w) => s + w.tokens.length, 0) / samples.length
    : 0

  const mergeEfficiency = stats.total_merges > 0
    ? (stats.total_merges / (stats.vocab_size - stats.base_chars - stats.special_tokens)) * 100
    : 0

  return { vocabUtilization, compressionRatio, avgTokensPerWord, mergeEfficiency }
}

export function TokenizerEfficiencyCard({ stats, samples }: TokenizerEfficiencyCardProps) {
  const efficiency = computeEfficiency(stats, samples)

  if (!efficiency || !stats) return null

  return (
    <Card data-testid="tokenizer-efficiency">
      <CardHeader>
        <CardTitle className="text-base">Tokenizer Efficiency</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Vocab Utilization</div>
            <div className="text-lg font-semibold">{efficiency.vocabUtilization.toFixed(0)}%</div>
            <div className="text-[10px] text-muted-foreground">
              {efficiency.vocabUtilization > 80 ? 'High' : efficiency.vocabUtilization > 50 ? 'Medium' : 'Low'}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Compression</div>
            <div className="text-lg font-semibold">{efficiency.compressionRatio.toFixed(1)}x</div>
            <div className="text-[10px] text-muted-foreground">chars per token</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Avg Tokens/Word</div>
            <div className="text-lg font-semibold">{efficiency.avgTokensPerWord.toFixed(1)}</div>
            <div className="text-[10px] text-muted-foreground">
              {efficiency.avgTokensPerWord <= 1.5 ? 'Efficient' : efficiency.avgTokensPerWord <= 3 ? 'Average' : 'Fragmented'}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Merge Efficiency</div>
            <div className="text-lg font-semibold">{efficiency.mergeEfficiency.toFixed(0)}%</div>
            <div className="text-[10px] text-muted-foreground">merge coverage</div>
          </div>
        </div>
        <div className="mt-3 text-[11px] text-muted-foreground">
          {stats.special_tokens} special tokens · {stats.total_merges} merges learned
        </div>
      </CardContent>
    </Card>
  )
}
