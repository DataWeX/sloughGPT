'use client'

import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'

const PHONEME_CHART: { symbol: string; ipa: string; example: string; description: string }[] = [
  { symbol: 'P', ipa: '/p/', example: 'pat', description: 'voiceless bilabial stop' },
  { symbol: 'B', ipa: '/b/', example: 'bat', description: 'voiced bilabial stop' },
  { symbol: 'T', ipa: '/t/', example: 'tap', description: 'voiceless alveolar stop' },
  { symbol: 'D', ipa: '/d/', example: 'dab', description: 'voiced alveolar stop' },
  { symbol: 'K', ipa: '/k/', example: 'cat', description: 'voiceless velar stop' },
  { symbol: 'G', ipa: '/g/', example: 'gap', description: 'voiced velar stop' },
  { symbol: 'F', ipa: '/f/', example: 'fat', description: 'voiceless labiodental fricative' },
  { symbol: 'V', ipa: '/v/', example: 'vat', description: 'voiced labiodental fricative' },
  { symbol: 'TH', ipa: '/θ/', example: 'think', description: 'voiceless dental fricative' },
  { symbol: 'DH', ipa: '/ð/', example: 'this', description: 'voiced dental fricative' },
  { symbol: 'S', ipa: '/s/', example: 'sit', description: 'voiceless alveolar fricative' },
  { symbol: 'Z', ipa: '/z/', example: 'zip', description: 'voiced alveolar fricative' },
  { symbol: 'SH', ipa: '/ʃ/', example: 'ship', description: 'voiceless postalveolar fricative' },
  { symbol: 'ZH', ipa: '/ʒ/', example: 'measure', description: 'voiced postalveolar fricative' },
  { symbol: 'CH', ipa: '/tʃ/', example: 'chip', description: 'voiceless postalveolar affricate' },
  { symbol: 'JH', ipa: '/dʒ/', example: 'job', description: 'voiced postalveolar affricate' },
  { symbol: 'M', ipa: '/m/', example: 'mat', description: 'bilabial nasal' },
  { symbol: 'N', ipa: '/n/', example: 'nap', description: 'alveolar nasal' },
  { symbol: 'NG', ipa: '/ŋ/', example: 'sing', description: 'velar nasal' },
  { symbol: 'L', ipa: '/l/', example: 'lap', description: 'alveolar lateral approximant' },
  { symbol: 'R', ipa: '/ɹ/', example: 'rap', description: 'alveolar approximant' },
  { symbol: 'W', ipa: '/w/', example: 'wet', description: 'labial-velar approximant' },
  { symbol: 'Y', ipa: '/j/', example: 'yet', description: 'palatal approximant' },
  { symbol: 'HH', ipa: '/h/', example: 'hat', description: 'glottal fricative' },
  { symbol: 'AE', ipa: '/æ/', example: 'cat', description: 'near-open front unrounded' },
  { symbol: 'AH', ipa: '/ʌ/', example: 'cut', description: 'open-mid back unrounded' },
  { symbol: 'AO', ipa: '/ɔ/', example: 'caught', description: 'open-mid back rounded' },
  { symbol: 'OW', ipa: '/oʊ/', example: 'go', description: 'close-mid back rounded diphthong' },
  { symbol: 'UH', ipa: '/ʊ/', example: 'put', description: 'near-close near-back rounded' },
  { symbol: 'UW', ipa: '/u/', example: 'food', description: 'close back rounded' },
  { symbol: 'EH', ipa: '/ɛ/', example: 'bed', description: 'open-mid front unrounded' },
  { symbol: 'ER', ipa: '/ɝ/', example: 'bird', description: 'rhotacized mid central' },
  { symbol: 'AY', ipa: '/aɪ/', example: 'hide', description: 'open front unrounded diphthong' },
  { symbol: 'AW', ipa: '/aʊ/', example: 'how', description: 'open front unrounded diphthong' },
  { symbol: 'OY', ipa: '/ɔɪ/', example: 'boy', description: 'open-mid back rounded diphthong' },
  { symbol: 'IH', ipa: '/ɪ/', example: 'sit', description: 'near-close near-front unrounded' },
  { symbol: 'IY', ipa: '/i/', example: 'see', description: 'close front unrounded' },
  { symbol: 'AX', ipa: '/ə/', example: 'about', description: 'mid central (schwa)' },
]

function groupByCategory(phonemes: typeof PHONEME_CHART) {
  const stops = phonemes.filter(p => ['P', 'B', 'T', 'D', 'K', 'G'].includes(p.symbol))
  const fricatives = phonemes.filter(p => ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH'].includes(p.symbol))
  const affricates = phonemes.filter(p => ['CH', 'JH'].includes(p.symbol))
  const nasals = phonemes.filter(p => ['M', 'N', 'NG'].includes(p.symbol))
  const approximants = phonemes.filter(p => ['L', 'R', 'W', 'Y'].includes(p.symbol))
  const vowels = phonemes.filter(p => ['AE', 'AH', 'AO', 'OW', 'UH', 'UW', 'EH', 'ER', 'AY', 'AW', 'OY', 'IH', 'IY', 'AX'].includes(p.symbol))
  return { stops, fricatives, affricates, nasals, approximants, vowels }
}

function PhonemeGroup({ title, phonemes }: { title: string; phonemes: typeof PHONEME_CHART }) {
  if (phonemes.length === 0) return null
  return (
    <div>
      <p className="text-xs font-medium text-muted-foreground mb-1">{title}</p>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-1">
        {phonemes.map(p => (
          <div key={p.symbol} className="flex items-center gap-2 p-1.5 rounded bg-background text-xs">
            <Badge variant="outline" className="w-10 justify-center font-mono">{p.symbol}</Badge>
            <div className="min-w-0">
              <span className="font-mono text-muted-foreground">{p.ipa}</span>
              <span className="text-muted-foreground ml-1">{p.example}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function PhonemeReference() {
  const groups = groupByCategory(PHONEME_CHART)

  return (
    <Collapsible>
      <Card>
        <CollapsibleTrigger className="w-full">
          <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors rounded-lg">
            <CardTitle className="flex items-center justify-between text-base">
              <span>Phoneme Reference</span>
              <span className="text-xs font-normal text-muted-foreground">{PHONEME_CHART.length} symbols</span>
            </CardTitle>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="space-y-4 pt-0">
            <PhonemeGroup title="Stops" phonemes={groups.stops} />
            <PhonemeGroup title="Fricatives" phonemes={groups.fricatives} />
            <PhonemeGroup title="Affricates" phonemes={groups.affricates} />
            <PhonemeGroup title="Nasals" phonemes={groups.nasals} />
            <PhonemeGroup title="Approximants" phonemes={groups.approximants} />
            <PhonemeGroup title="Vowels & Diphthongs" phonemes={groups.vowels} />
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}
