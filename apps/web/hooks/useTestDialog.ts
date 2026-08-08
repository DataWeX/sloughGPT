'use client'

import { useState, useCallback } from 'react'
import { generateController } from '@/lib/generate-controller'
import { extractErrorMessage } from '@/lib/error-utils'

export interface TestModelResult {
  prompt: string
  response: string
  model: string
  tokens_generated: number
  error: string
}

export interface UseTestDialogReturn {
  testDialogOpen: boolean
  testPrompt: string
  testResult: TestModelResult | null
  testLoading: boolean
  setTestDialogOpen: (open: boolean) => void
  setTestPrompt: (prompt: string) => void
  handleTestModel: () => Promise<void>
  clearTest: () => void
}

export function useTestDialog(): UseTestDialogReturn {
  const [testDialogOpen, setTestDialogOpen] = useState(false)
  const [testPrompt, setTestPrompt] = useState('')
  const [testResult, setTestResult] = useState<TestModelResult | null>(null)
  const [testLoading, setTestLoading] = useState(false)

  const handleTestModel = useCallback(async () => {
    if (!testPrompt.trim()) return
    setTestLoading(true)
    setTestResult(null)
    try {
      const data = await generateController.generate({
        prompt: testPrompt,
        max_new_tokens: 100,
        temperature: 0.8,
      })
      setTestResult({
        prompt: testPrompt,
        response: data.text || '(empty)',
        model: data.model || '',
        tokens_generated: data.tokens_generated || 0,
        error: '',
      })
    } catch (e) {
      setTestResult({
        prompt: testPrompt,
        response: '',
        model: '',
        tokens_generated: 0,
        error: extractErrorMessage(e, 'unknown error'),
      })
    } finally { setTestLoading(false) }
  }, [testPrompt])

  const clearTest = useCallback(() => {
    setTestPrompt(''); setTestResult(null)
  }, [])

  return {
    testDialogOpen, testPrompt, testResult, testLoading,
    setTestDialogOpen, setTestPrompt, handleTestModel, clearTest,
  }
}
