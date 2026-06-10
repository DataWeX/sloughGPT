'use client'

import { useState, useCallback } from 'react'
import { PUBLIC_API_URL } from '@/lib/config'

export interface UseTestDialogReturn {
  testDialogOpen: boolean
  testPrompt: string
  testOutput: string
  testLoading: boolean
  setTestDialogOpen: (open: boolean) => void
  setTestPrompt: (prompt: string) => void
  setTestOutput: (output: string) => void
  setTestLoading: (loading: boolean) => void
  handleTestModel: () => Promise<void>
  clearTest: () => void
}

export function useTestDialog(): UseTestDialogReturn {
  const [testDialogOpen, setTestDialogOpen] = useState(false)
  const [testPrompt, setTestPrompt] = useState('')
  const [testOutput, setTestOutput] = useState('')
  const [testLoading, setTestLoading] = useState(false)

  const handleTestModel = useCallback(async () => {
    if (!testPrompt.trim()) return
    setTestLoading(true)
    setTestOutput('')
    try {
      const res = await fetch(`${PUBLIC_API_URL}/inference/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: testPrompt, max_new_tokens: 100, temperature: 0.8 }),
      })
      if (!res.ok) throw new Error('Inference failed')
      const data = await res.json()
      setTestOutput(data.text || '(empty)')
    } catch (e) {
      setTestOutput(`Error: ${e instanceof Error ? e.message : 'unknown'}`)
    } finally { setTestLoading(false) }
  }, [testPrompt])

  const clearTest = useCallback(() => {
    setTestPrompt(''); setTestOutput('')
  }, [])

  return {
    testDialogOpen, testPrompt, testOutput, testLoading,
    setTestDialogOpen, setTestPrompt, setTestOutput, setTestLoading,
    handleTestModel, clearTest,
  }
}
