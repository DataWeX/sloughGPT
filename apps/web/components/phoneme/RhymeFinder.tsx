'use client'

import { useState, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, toIPA, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { useToastStore } from '@/lib/toast-store'
import PhonemeSkeleton from './PhonemeSkeleton'

interface RhymeResult {
  word: string
  phonemes: string[]
  rhymePart: string
  matchType: 'perfect' | 'near' | 'slant'
}

const RHYME_DATABASE: Record<string, { word: string; phonemes: string[] }[]> = {
  'at': [
    { word: 'cat', phonemes: ['K', 'AE', 'T'] },
    { word: 'hat', phonemes: ['HH', 'AE', 'T'] },
    { word: 'bat', phonemes: ['B', 'AE', 'T'] },
    { word: 'mat', phonemes: ['M', 'AE', 'T'] },
    { word: 'rat', phonemes: ['R', 'AE', 'T'] },
    { word: 'sat', phonemes: ['S', 'AE', 'T'] },
    { word: 'fat', phonemes: ['F', 'AE', 'T'] },
    { word: 'flat', phonemes: ['F', 'L', 'AE', 'T'] },
    { word: 'that', phonemes: ['DH', 'AE', 'T'] },
    { word: 'plat', phonemes: ['P', 'L', 'AE', 'T'] },
  ],
  'et': [
    { word: 'bet', phonemes: ['B', 'EH', 'T'] },
    { word: 'get', phonemes: ['G', 'EH', 'T'] },
    { word: 'jet', phonemes: ['JH', 'EH', 'T'] },
    { word: 'let', phonemes: ['L', 'EH', 'T'] },
    { word: 'met', phonemes: ['M', 'EH', 'T'] },
    { word: 'net', phonemes: ['N', 'EH', 'T'] },
    { word: 'pet', phonemes: ['P', 'EH', 'T'] },
    { word: 'set', phonemes: ['S', 'EH', 'T'] },
    { word: 'wet', phonemes: ['W', 'EH', 'T'] },
    { word: 'yet', phonemes: ['Y', 'EH', 'T'] },
  ],
  'it': [
    { word: 'bit', phonemes: ['B', 'IH', 'T'] },
    { word: 'fit', phonemes: ['F', 'IH', 'T'] },
    { word: 'hit', phonemes: ['HH', 'IH', 'T'] },
    { word: 'kit', phonemes: ['K', 'IH', 'T'] },
    { word: 'lit', phonemes: ['L', 'IH', 'T'] },
    { word: 'mit', phonemes: ['M', 'IH', 'T'] },
    { word: 'pit', phonemes: ['P', 'IH', 'T'] },
    { word: 'sit', phonemes: ['S', 'IH', 'T'] },
    { word: 'wit', phonemes: ['W', 'IH', 'T'] },
    { word: 'split', phonemes: ['S', 'P', 'L', 'IH', 'T'] },
  ],
  'ot': [
    { word: 'bot', phonemes: ['B', 'AO', 'T'] },
    { word: 'cot', phonemes: ['K', 'AO', 'T'] },
    { word: 'dot', phonemes: ['D', 'AO', 'T'] },
    { word: 'got', phonemes: ['G', 'AO', 'T'] },
    { word: 'hot', phonemes: ['HH', 'AO', 'T'] },
    { word: 'jot', phonemes: ['JH', 'AO', 'T'] },
    { word: 'lot', phonemes: ['L', 'AO', 'T'] },
    { word: 'not', phonemes: ['N', 'AO', 'T'] },
    { word: 'pot', phonemes: ['P', 'AO', 'T'] },
    { word: 'shot', phonemes: ['SH', 'AO', 'T'] },
  ],
  'ut': [
    { word: 'but', phonemes: ['B', 'AH', 'T'] },
    { word: 'cut', phonemes: ['K', 'AH', 'T'] },
    { word: 'gut', phonemes: ['G', 'AH', 'T'] },
    { word: 'hut', phonemes: ['HH', 'AH', 'T'] },
    { word: 'nut', phonemes: ['N', 'AH', 'T'] },
    { word: 'put', phonemes: ['P', 'UH', 'T'] },
    { word: 'rut', phonemes: ['R', 'AH', 'T'] },
    { word: 'shut', phonemes: ['SH', 'AH', 'T'] },
    { word: 'strut', phonemes: ['S', 'T', 'R', 'AH', 'T'] },
  ],
  'ake': [
    { word: 'bake', phonemes: ['B', 'EY', 'K'] },
    { word: 'cake', phonemes: ['K', 'EY', 'K'] },
    { word: 'fake', phonemes: ['F', 'EY', 'K'] },
    { word: 'lake', phonemes: ['L', 'EY', 'K'] },
    { word: 'make', phonemes: ['M', 'EY', 'K'] },
    { word: 'rake', phonemes: ['R', 'EY', 'K'] },
    { word: 'sake', phonemes: ['S', 'EY', 'K'] },
    { word: 'take', phonemes: ['T', 'EY', 'K'] },
    { word: 'wake', phonemes: ['W', 'EY', 'K'] },
    { word: 'snake', phonemes: ['S', 'N', 'EY', 'K'] },
  ],
  'ight': [
    { word: 'bite', phonemes: ['B', 'AY', 'T'] },
    { word: 'kite', phonemes: ['K', 'AY', 'T'] },
    { word: 'light', phonemes: ['L', 'AY', 'T'] },
    { word: 'might', phonemes: ['M', 'AY', 'T'] },
    { word: 'night', phonemes: ['N', 'AY', 'T'] },
    { word: 'right', phonemes: ['R', 'AY', 'T'] },
    { word: 'sight', phonemes: ['S', 'AY', 'T'] },
    { word: 'tight', phonemes: ['T', 'AY', 'T'] },
    { word: 'white', phonemes: ['W', 'AY', 'T'] },
    { word: 'flight', phonemes: ['F', 'L', 'AY', 'T'] },
  ],
  'eel': [
    { word: 'feel', phonemes: ['F', 'IY', 'L'] },
    { word: 'heal', phonemes: ['HH', 'IY', 'L'] },
    { word: 'keel', phonemes: ['K', 'IY', 'L'] },
    { word: 'meal', phonemes: ['M', 'IY', 'L'] },
    { word: 'peel', phonemes: ['P', 'IY', 'L'] },
    { word: 'real', phonemes: ['R', 'IY', 'L'] },
    { word: 'seal', phonemes: ['S', 'IY', 'L'] },
    { word: 'steal', phonemes: ['S', 'T', 'IY', 'L'] },
    { word: 'wheel', phonemes: ['W', 'IY', 'L'] },
    { word: 'zeal', phonemes: ['Z', 'IY', 'L'] },
  ],
  'an': [
    { word: 'ban', phonemes: ['B', 'AE', 'N'] },
    { word: 'can', phonemes: ['K', 'AE', 'N'] },
    { word: 'fan', phonemes: ['F', 'AE', 'N'] },
    { word: 'man', phonemes: ['M', 'AE', 'N'] },
    { word: 'pan', phonemes: ['P', 'AE', 'N'] },
    { word: 'ran', phonemes: ['R', 'AE', 'N'] },
    { word: 'tan', phonemes: ['T', 'AE', 'N'] },
    { word: 'van', phonemes: ['V', 'AE', 'N'] },
    { word: 'plan', phonemes: ['P', 'L', 'AE', 'N'] },
    { word: 'scan', phonemes: ['S', 'K', 'AE', 'N'] },
  ],
  'ore': [
    { word: 'core', phonemes: ['K', 'AO', 'R'] },
    { word: 'door', phonemes: ['D', 'AO', 'R'] },
    { word: 'floor', phonemes: ['F', 'L', 'AO', 'R'] },
    { word: 'more', phonemes: ['M', 'AO', 'R'] },
    { word: 'pour', phonemes: ['P', 'AO', 'R'] },
    { word: 'shore', phonemes: ['SH', 'AO', 'R'] },
    { word: 'snore', phonemes: ['S', 'N', 'AO', 'R'] },
    { word: 'store', phonemes: ['S', 'T', 'AO', 'R'] },
    { word: 'sure', phonemes: ['SH', 'UH', 'R'] },
    { word: 'war', phonemes: ['W', 'AO', 'R'] },
  ],
}

function getRhymeKey(phonemes: string[]): string {
  // Get the last 2-3 phonemes as the rhyme key
  if (phonemes.length < 2) return phonemes.join('')
  return phonemes.slice(-2).join('')
}

function findRhymes(inputPhonemes: string[], language: PhonemeLanguage): RhymeResult[] {
  const inputKey = getRhymeKey(inputPhonemes)
  const results: RhymeResult[] = []

  // Search through all rhyme groups
  for (const [, words] of Object.entries(RHYME_DATABASE)) {
    for (const entry of words) {
      const entryKey = getRhymeKey(entry.phonemes)
      if (entryKey === inputKey) {
        results.push({
          word: entry.word,
          phonemes: entry.phonemes,
          rhymePart: entryKey,
          matchType: 'perfect',
        })
      } else if (entryKey.slice(-1) === inputKey.slice(-1)) {
        results.push({
          word: entry.word,
          phonemes: entry.phonemes,
          rhymePart: entryKey,
          matchType: 'near',
        })
      }
    }
  }

  return results.slice(0, 12)
}

export default function RhymeFinder() {
  const [word, setWord] = useState('')
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<RhymeResult[]>([])
  const [inputPhonemes, setInputPhonemes] = useState<string[]>([])
  const addToast = useToastStore(s => s.addToast)

  const handleSearch = useCallback(async () => {
    if (!word.trim()) return
    setLoading(true)
    try {
      const res = await phonemeController.encode(word, language)
      setInputPhonemes(res.phonemes)
      const rhymes = findRhymes(res.phonemes, language)
      setResults(rhymes)
      if (rhymes.length === 0) {
        addToast('No rhymes found', 'info')
      }
    } catch {
      addToast('Encoding failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [word, language, addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Rhyme Finder</span>
          {results.length > 0 && (
            <Badge variant="outline">{results.length} rhymes</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <Input
            value={word}
            onChange={e => setWord(e.target.value)}
            placeholder="Enter a word to find rhymes..."
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            className="flex-1"
          />
          <Select value={language} onValueChange={v => setLanguage(v as PhonemeLanguage)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PHONEME_LANGUAGES.map(l => (
                <SelectItem key={l.value} value={l.value}>{l.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button onClick={handleSearch} disabled={loading || !word.trim()}>
            {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
            Find Rhymes
          </Button>
        </div>

        {loading && <PhonemeSkeleton variant="encode" />}

        {!loading && inputPhonemes.length > 0 && (
          <div className="p-3 rounded-lg bg-muted/50">
            <p className="text-sm text-muted-foreground mb-1">Phonemes for &quot;{word}&quot;:</p>
            <div className="flex flex-wrap gap-1">
              {inputPhonemes.map((p, i) => (
                <Badge key={i} variant="outline">{p}</Badge>
              ))}
            </div>
          </div>
        )}

        {!loading && results.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-medium text-muted-foreground">Rhyming words:</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {results.map((result, i) => (
                <div
                  key={i}
                  className="p-2 rounded-lg bg-muted/30 text-center"
                >
                  <p className="font-medium">{result.word}</p>
                  <div className="flex flex-wrap gap-1 justify-center mt-1">
                    {result.phonemes.map((p, j) => (
                      <Badge key={j} variant="outline" className="text-xs">{p}</Badge>
                    ))}
                  </div>
                  <Badge
                    variant={result.matchType === 'perfect' ? 'default' : 'secondary'}
                    className="mt-1 text-xs"
                  >
                    {result.matchType}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && word && results.length === 0 && inputPhonemes.length > 0 && (
          <p className="text-muted-foreground text-center py-4">
            No rhymes found for &quot;{word}&quot;
          </p>
        )}
      </CardContent>
    </Card>
  )
}
