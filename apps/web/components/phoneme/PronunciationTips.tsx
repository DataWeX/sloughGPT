'use client'

import { useState, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { PHONEME_LANGUAGES } from '@/lib/phoneme-controller'
import type { PhonemeLanguage } from '@/lib/phoneme-controller'

interface Tip {
  phoneme: string
  ipa: string
  description: string
  commonMistakes: string
  practiceWords: string[]
}

const TIPS: Record<PhonemeLanguage, Tip[]> = {
  en: [
    { phoneme: 'TH', ipa: 'θ', description: 'Place tongue between teeth, blow air gently. Not a "T" or "F".', commonMistakes: 'Often replaced with T, F, or D', practiceWords: ['think', 'three', 'bath', 'math'] },
    { phoneme: 'DH', ipa: 'ð', description: 'Voiced version of TH. Tongue between teeth with vocal cord vibration.', commonMistakes: 'Often replaced with D or Z', practiceWords: ['this', 'that', 'the', 'breathe'] },
    { phoneme: 'R', ipa: 'ɹ', description: 'Curl tongue back without touching roof of mouth. Lips slightly rounded.', commonMistakes: 'British R vs American R confusion', practiceWords: ['run', 'red', 'car', 'water'] },
    { phoneme: 'SH', ipa: 'ʃ', description: 'Round lips, push tongue forward. Like telling someone to be quiet.', commonMistakes: 'Confused with S or CH', practiceWords: ['ship', 'she', 'wish', 'nation'] },
    { phoneme: 'AE', ipa: 'æ', description: 'Open mouth wide, tongue low and front. Between "ah" and "eh".', commonMistakes: 'Confused with EH or AH', practiceWords: ['cat', 'bad', 'man', 'happy'] },
  ],
  de: [
    { phoneme: 'CH', ipa: 'ç/x', description: 'After front vowels: light "hiss". After back vowels: deep "scrape".', commonMistakes: 'English speakers use SH instead', practiceWords: ['ich', 'acht', 'Buch', 'milch'] },
    { phoneme: 'UE', ipa: 'yː', description: 'Round lips like saying "oo" but say "ee".', commonMistakes: 'Confused with regular U', practiceWords: ['grün', 'Tür', 'über', 'fünf'] },
    { phoneme: 'OE', ipa: 'øː', description: 'Round lips like saying "oo" but say "ay".', commonMistakes: 'Confused with regular O', practiceWords: ['schön', 'Möbel', 'Köln', 'böse'] },
    { phoneme: 'R', ipa: 'ʁ', description: 'Throat R, like a gentle gargle. uvular trill.', commonMistakes: 'English R is wrong here', practiceWords: ['Rot', 'Rad', 'Tier', 'Haus'] },
    { phoneme: 'Z', ipa: 'ts', description: 'Like "ts" in "cats". Voiceless affricate.', commonMistakes: 'Confused with English Z (which is voiced)', practiceWords: ['Zeit', 'zwei', 'Herz', 'Blitz'] },
  ],
  fr: [
    { phoneme: 'R', ipa: 'ʁ', description: 'Guttural R from back of throat. uvular fricative.', commonMistakes: 'English speakers use tongue R', practiceWords: ['rouge', 'rire', 'Paris', 'mer'] },
    { phoneme: 'U', ipa: 'y', description: 'Round lips like "oo" but say "ee". Like German Ü.', commonMistakes: 'Confused with English U', practiceWords: ['tu', 'vu', 'su', 'lu'] },
    { phoneme: 'ON', ipa: 'ɔ̃', description: 'Nasal O. Say "oh" through nose, no "n" sound.', commonMistakes: 'Adding an N sound', practiceWords: ['bon', 'mon', 'son', 'front'] },
    { phoneme: 'AN', ipa: 'ɑ̃', description: 'Nasal A. Say "ah" through nose, no "n" sound.', commonMistakes: 'Adding an N sound', practiceWords: ['dans', 'vent', 'temps', 'enfant'] },
    { phoneme: 'EU', ipa: 'ø', description: 'Say "uh" with rounded lips.', commonMistakes: 'Confused with regular E', practiceWords: ['feu', 'peu', 'deux', 'bleu'] },
  ],
  es: [
    { phoneme: 'RR', ipa: 'r', description: 'Strong rolled R with tongue trill.', commonMistakes: 'Using English R instead', practiceWords: ['perro', 'carro', 'rrico', 'tierra'] },
    { phoneme: 'Ñ', ipa: 'ɲ', description: 'Like "ny" in canyon. Palatal nasal.', commonMistakes: 'Confused with N', practiceWords: ['año', 'niño', 'señor', 'español'] },
    { phoneme: 'LL', ipa: 'ʎ', description: 'Palatal lateral. Like "lli" in million.', commonMistakes: 'Regional variations (yeísmo)', practiceWords: ['calle', 'llegar', 'lluvia', 'pollo'] },
    { phoneme: 'J', ipa: 'x', description: 'Deep throat H sound. Like Scottish "loch".', commonMistakes: 'Using English J', practiceWords: ['jamón', 'jardín', 'jefe', 'reloj'] },
    { phoneme: 'B/V', ipa: 'β', description: 'Softer than English B. Lips barely touch.', commonMistakes: 'Hard B or V distinction', practiceWords: ['vino', 'bueno', 'verde', 'ibuprofeno'] },
  ],
  it: [
    { phoneme: 'GL', ipa: 'ʎ', description: 'Palatal lateral. Tongue touches roof near sides.', commonMistakes: 'Confused with G+L', practiceWords: ['figlio', 'giglio', 'biglietto', 'ogli'] },
    { phoneme: 'GN', ipa: 'ɲ', description: 'Like "ny" in canyon. Palatal nasal.', commonMistakes: 'Confused with G+N', practiceWords: ['gnomo', 'gnocchi', 'compagno', 'ogni'] },
    { phoneme: 'SC', ipa: 'sk', description: 'Always pronounced as S+K, never SH.', commonMistakes: 'Pronouncing as SH', practiceWords: ['scuola', 'pesce', 'scusa', 'musica'] },
    { phoneme: 'C', ipa: 'k/tʃ', description: 'Before A/O/U: hard K. Before E/I: soft CH.', commonMistakes: 'Always hard or always soft', practiceWords: ['casa', 'cena', 'come', 'che'] },
    { phoneme: 'Q', ipa: 'kw', description: 'Always followed by U. Like "qu" in queen.', commonMistakes: 'Separating Q from U', practiceWords: ['qua', 'quando', 'qui', 'qualche'] },
  ],
  pt: [
    { phoneme: 'R', ipa: 'ʁ/h', description: 'At start: guttural H. Between vowels: flap.', commonMistakes: 'Using English R', practiceWords: ['Rio', 'cafezinho', 'caro', 'para'] },
    { phoneme: 'NH', ipa: 'ɲ', description: 'Like "ny" in canyon. Palatal nasal.', commonMistakes: 'Confused with N', practiceWords: ['manhã', 'banho', 'ninho', 'sonho'] },
    { phoneme: 'LH', ipa: 'ʎ', description: 'Palatal lateral. Like "lli" in million.', commonMistakes: 'Confused with L', practiceWords: ['filho', 'trabalho', 'olho', 'palhaço'] },
    { phoneme: 'ÃO', ipa: 'ɐ̃w̃', description: 'Nasal diphthong. Nasal "ow" sound.', commonMistakes: 'Non-native speakers struggle most', practiceWords: ['não', 'mão', 'pão', 'irmão'] },
    { phoneme: 'S', ipa: 's/z', description: 'Before consonant or at end: S. Between vowels: Z.', commonMistakes: 'Always pronouncing as S', practiceWords: ['casa', 'mesa', 'estrela', 'este'] },
  ],
}

export default function PronunciationTips() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const history = usePhonemeStore(s => s.history)

  const tips = useMemo(() => TIPS[language] || [], [language])

  const usedLangs = useMemo(() => {
    return [...new Set(history.map(e => e.language))]
  }, [history])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Tips</span>
          <Badge variant="outline">{tips.length} tips</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <Select value={language} onValueChange={v => setLanguage(v as PhonemeLanguage)}>
            <SelectTrigger className="w-[160px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PHONEME_LANGUAGES.map(lang => (
                <SelectItem key={lang.value} value={lang.value}>
                  {lang.label} {usedLangs.includes(lang.value as PhonemeLanguage) ? '✓' : ''}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {tips.map(tip => (
          <Collapsible key={tip.phoneme}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/30 transition-colors cursor-pointer text-sm">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="min-w-[50px] justify-center font-mono">{tip.phoneme}</Badge>
                  <span className="text-xs text-muted-foreground">{tip.ipa}</span>
                </div>
                <span className="text-xs text-muted-foreground">{tip.practiceWords.length} practice words</span>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-3 pt-0 space-y-2">
                <p className="text-sm">{tip.description}</p>
                <p className="text-xs text-destructive">Common mistake: {tip.commonMistakes}</p>
                <div className="flex flex-wrap gap-1">
                  {tip.practiceWords.map(word => (
                    <Badge key={word} variant="secondary" className="text-xs">{word}</Badge>
                  ))}
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  )
}
