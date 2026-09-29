'use client'

import { useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { toIPA } from '@/lib/phoneme-controller'

interface PhonemeKeyboardProps {
  onInsert: (phoneme: string) => void
  onBackspace: () => void
  onClear: () => void
}

const PHONEME_LAYOUT = [
  { row: 'stops', phonemes: ['P', 'B', 'T', 'D', 'K', 'G'] },
  { row: 'fricatives', phonemes: ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH'] },
  { row: 'affricates', phonemes: ['CH', 'JH'] },
  { row: 'nasals', phonemes: ['M', 'N', 'NG'] },
  { row: 'liquids', phonemes: ['L', 'R'] },
  { row: 'glides', phonemes: ['W', 'Y'] },
  {
    row: 'vowels',
    phonemes: ['IY', 'IH', 'EY', 'EH', 'AE', 'AA', 'AH', 'AO', 'OW', 'OY', 'UH', 'UW', 'ER', 'AX'],
  },
]

function getPhonemeColor(phoneme: string): string {
  const stops = ['P', 'B', 'T', 'D', 'K', 'G']
  const fricatives = ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH']
  const affricates = ['CH', 'JH']
  const nasals = ['M', 'N', 'NG']
  const liquids = ['L', 'R']
  const glides = ['W', 'Y']

  if (stops.includes(phoneme)) return 'bg-info/15 hover:bg-info text-info border-info'
  if (fricatives.includes(phoneme))
    return 'bg-primary/15 hover:bg-primary text-primary border-primary'
  if (affricates.includes(phoneme)) return 'bg-chart-5 hover:bg-chart-5 text-chart-5 border-chart-5'
  if (nasals.includes(phoneme)) return 'bg-success/15 hover:bg-success text-success border-success'
  if (liquids.includes(phoneme)) return 'bg-accent hover:bg-accent text-accent border-accent'
  if (glides.includes(phoneme)) return 'bg-info hover:bg-info text-info border-info'
  return 'bg-gray-100 hover:bg-gray-200 text-gray-800 border-gray-300'
}

export default function PhonemeKeyboard({ onInsert, onBackspace, onClear }: PhonemeKeyboardProps) {
  const handlePhonemeClick = useCallback(
    (phoneme: string) => {
      onInsert(phoneme)
    },
    [onInsert],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Phoneme Keyboard</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={onBackspace}>
              ⌫
            </Button>
            <Button variant="outline" size="sm" onClick={onClear}>
              Clear
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {PHONEME_LAYOUT.map((row) => (
          <div key={row.row} className="flex flex-wrap gap-1">
            {row.phonemes.map((phoneme) => {
              const ipa = toIPA([phoneme])[0] || phoneme
              return (
                <Button
                  key={phoneme}
                  variant="outline"
                  size="sm"
                  className={`h-8 px-2 text-xs font-mono ${getPhonemeColor(phoneme)}`}
                  onClick={() => handlePhonemeClick(phoneme)}
                >
                  <span className="flex flex-col items-center">
                    <span>{phoneme}</span>
                    <span className="text-[10px] opacity-70">{ipa}</span>
                  </span>
                </Button>
              )
            })}
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
