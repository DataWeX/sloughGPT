'use client'

import { useState, useCallback, useRef } from 'react'
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
  testStreaming: boolean
  testStreamingText: string
  responseFormat: 'text' | 'json'
  setTestDialogOpen: (open: boolean) => void
  setTestPrompt: (prompt: string) => void
  setResponseFormat: (format: 'text' | 'json') => void
  handleTestModel: () => Promise<void>
  clearTest: () => void
}

export function useTestDialog(): UseTestDialogReturn {
  const [testDialogOpen, setTestDialogOpen] = useState(false)
  const [testPrompt, setTestPrompt] = useState('')
  const [testResult, setTestResult] = useState<TestModelResult | null>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [testStreaming, setTestStreaming] = useState(false)
  const [testStreamingText, setTestStreamingText] = useState('')
  const [responseFormat, setResponseFormat] = useState<'text' | 'json'>('text')
  const streamingRef = useRef(false)

  const handleTestModel = useCallback(async () => {
    if (!testPrompt.trim()) return
    setTestLoading(true)
    setTestStreaming(true)
    setTestStreamingText('')
    setTestResult(null)
    streamingRef.current = true

    let accumulated = ''
    let model = ''
    let tokensGenerated = 0

    try {
      await generateController.generateStream(
        {
          prompt: testPrompt,
          max_new_tokens: 256,
          temperature: 0.8,
          response_format: responseFormat,
        },
        (token) => {
          if (!streamingRef.current) return
          accumulated += token
          setTestStreamingText(accumulated)
        },
        () => {
          if (!streamingRef.current) return
          setTestResult({
            prompt: testPrompt,
            response: accumulated || '(empty)',
            model,
            tokens_generated: tokensGenerated,
            error: '',
          })
        },
        (error) => {
          if (!streamingRef.current) return
          setTestResult({
            prompt: testPrompt,
            response: '',
            model: '',
            tokens_generated: 0,
            error,
          })
        },
      )
    } catch (e) {
      setTestResult({
        prompt: testPrompt,
        response: '',
        model: '',
        tokens_generated: 0,
        error: extractErrorMessage(e, 'unknown error'),
      })
    } finally {
      streamingRef.current = false
      setTestLoading(false)
      setTestStreaming(false)
    }
  }, [testPrompt, responseFormat])

  const clearTest = useCallback(() => {
    setTestPrompt('')
    setTestResult(null)
    setTestStreamingText('')
    setResponseFormat('text')
  }, [])

  return {
    testDialogOpen, testPrompt, testResult, testLoading,
    testStreaming, testStreamingText, responseFormat,
    setTestDialogOpen, setTestPrompt, setResponseFormat,
    handleTestModel, clearTest,
  }
}
