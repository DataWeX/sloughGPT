'use client'
import { notify } from '@/lib/notify'

import { useState, useCallback, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input } from '@sloughgpt/strui'
import { IconTrash } from '@sloughgpt/strui'
import {
  phonemeController,
  PHONEME_LANGUAGES,
  type PhonemeLanguage,
} from '@/lib/phoneme-controller'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

const STORAGE_KEY = 'sloughgpt-phoneme-custom-words'

interface CustomWord {
  id: string
  word: string
  language: PhonemeLanguage
  phonemes: string[]
  addedAt: number
}

function loadCustomWords(): CustomWord[] {
  if (typeof window === 'undefined') return []
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
  } catch {
    return []
  }
}

function saveCustomWords(words: CustomWord[]) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(words))
  } catch {
    /* best-effort */
  }
}

export default function CustomWordList() {
  const [words, setWords] = useState<CustomWord[]>([])
  const [newWord, setNewWord] = useState('')
  const [language, setLanguage] = useState<PhonemeLanguage>('en')

  useEffect(() => {
    setWords(loadCustomWords())
  }, [])

  const addWord = useCallback(async () => {
    const trimmed = newWord.trim().toLowerCase()
    if (!trimmed) return

    if (words.some((w) => w.word === trimmed && w.language === language)) {
      notify.toast(`"${trimmed}" already in your list`, { type: 'error' })
      return
    }

    try {
      const result = await phonemeController.encode(trimmed, language)
      const entry: CustomWord = {
        id: `${language}-${trimmed}-${Date.now()}`,
        word: trimmed,
        language,
        phonemes: result.phonemes,
        addedAt: Date.now(),
      }
      const updated = [...words, entry]
      setWords(updated)
      saveCustomWords(updated)
      setNewWord('')
      notify.toast(`Added "${trimmed}"`, { type: 'success' })
    } catch (err) {
      notify.toast(`Failed to encode "${trimmed}"`, { type: 'error' })
    }
  }, [newWord, language, words])

  const removeWord = useCallback(
    (id: string) => {
      const updated = words.filter((w) => w.id !== id)
      setWords(updated)
      saveCustomWords(updated)
      notify.toast('Word removed', { type: 'info' })
    },
    [words],
  )

  const clearAll = useCallback(() => {
    setWords([])
    saveCustomWords([])
    notify.toast('All custom words cleared', { type: 'info' })
  }, [])

  const groupedWords = words.reduce<Record<string, CustomWord[]>>((acc, w) => {
    acc[w.language] = acc[w.language] || []
    acc[w.language].push(w)
    return acc
  }, {})

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Custom Word List</span>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{words.length} words</Badge>
            {words.length > 0 && (
              <Button variant="ghost" size="sm" onClick={clearAll}>
                <IconTrash className="w-3 h-3" />
              </Button>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Select value={language} onValueChange={(v) => setLanguage(v as PhonemeLanguage)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PHONEME_LANGUAGES.map((lang) => (
                <SelectItem key={lang.value} value={lang.value}>
                  {lang.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            placeholder="Add a word to practice..."
            value={newWord}
            onChange={(e) => setNewWord(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addWord()}
            className="flex-1"
          />
          <Button onClick={addWord} disabled={!newWord.trim()}>
            Add
          </Button>
        </div>

        {Object.entries(groupedWords).map(([lang, langWords]) => (
          <div key={lang} className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="secondary">{lang.toUpperCase()}</Badge>
              <span className="text-xs text-muted-foreground">{langWords.length} words</span>
            </div>
            <div className="space-y-1">
              {langWords.map((w) => (
                <div key={w.id} className="flex items-center gap-2 p-2 rounded bg-muted/20 text-sm">
                  <span className="font-medium">{w.word}</span>
                  <Badge variant="outline" className="text-xs">
                    {w.phonemes.join('-')}
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="ml-auto h-6 w-6 p-0"
                    onClick={() => removeWord(w.id)}
                  >
                    <IconTrash className="w-3 h-3" />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        ))}

        {words.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-4">
            Add words you want to practice regularly. They will be saved locally.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
