'use client'

import { useState, useCallback, useRef } from 'react'
import { SoulNetWebGPU, SoulTransformerWebGPU, inferArch } from '@/lib/soulnet-webgpu'
import { PUBLIC_API_URL } from '@/lib/config'
import { extractErrorMessage } from '@/lib/error-utils'
import { devDebug } from '@/lib/dev-log'

export function useChatLocalEngine(
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void,
) {
  const [useLocalEngine, setUseLocalEngine] = useState(false)
  const [localEngineLoading, setLocalEngineLoading] = useState(false)
  const [localArchInfo, setLocalArchInfo] = useState<string | null>(null)
  const [localModelUrl, setLocalModelUrl] = useState('')
  const engineRef = useRef<SoulNetWebGPU | SoulTransformerWebGPU | null>(null)
  const engineLoadingRef = useRef(false)

  const initLocalEngine = useCallback(async (): Promise<boolean> => {
    if (engineRef.current || engineLoadingRef.current) return true
    if (!navigator.gpu) {
      showToast('WebGPU not available in this browser', 'error')
      devDebug('WebGPU unavailable', { navigator_gpu: false })
      return false
    }
    engineLoadingRef.current = true
    setLocalEngineLoading(true)
    try {
      if (!localModelUrl) throw new Error('No .soul file URL configured')
      const url = localModelUrl.startsWith('/auto-train/') || localModelUrl.startsWith('/sou/')
        ? `${PUBLIC_API_URL}${localModelUrl}`
        : localModelUrl
      devDebug('Fetching model for local engine', { url })
      const resp = await fetch(url)
      if (!resp.ok) throw new Error(`HTTP ${resp.status} from ${url}`)
      const buf = await resp.arrayBuffer()
      if (buf.byteLength > 500 * 1024 * 1024) {
        throw new Error(`Model too large for browser (${(buf.byteLength / 1024 / 1024).toFixed(0)}MB). Max: 500MB`)
      }
      devDebug('Model fetched', { size_bytes: buf.byteLength })
      const arch = inferArch(buf)
      devDebug('Inferred architecture', arch)

      if (arch.archType === 'transformer') {
        const engine = new SoulTransformerWebGPU()
        await engine.init()
        const embedDim = arch.embedDim
        const numLayers = arch.numLayers
        await engine.load(buf, {
          archType: 'transformer',
          embedDim,
          numHeads: 8,
          numKVHeads: 8,
          numLayers,
          dimFF: 1024,
          vocabSize: arch.vocabSize,
          maxSeqLen: 2048,
          eps: 1e-5,
        })
        engineRef.current = engine
        setLocalArchInfo(`${embedDim}×${numLayers}×8 Transformer`)
        showToast(`On-device AI ready (${embedDim}×${numLayers}L)`)
      } else {
        const engine = new SoulNetWebGPU()
        await engine.init()
        await engine.load(buf, { ...arch })
        engineRef.current = engine
        setLocalArchInfo(`${arch.embedDim}×${arch.hiddenDim} LSTM`)
        showToast(`Local AI ready (${arch.embedDim}x${arch.hiddenDim})`)
      }
      return true
    } catch (err) {
      const msg = extractErrorMessage(err, 'unknown error')
      showToast(`Failed to load local AI: ${msg}`, 'error')
      devDebug('Local engine init failed', { error: msg, url: localModelUrl })
      return false
    } finally {
      engineLoadingRef.current = false
      setLocalEngineLoading(false)
    }
  }, [showToast, localModelUrl])

  const handleToggleLocalEngine = useCallback(async () => {
    if (useLocalEngine) {
      setUseLocalEngine(false)
      setLocalArchInfo(null)
      showToast('Switched to server mode', 'info')
      devDebug('Switched to server mode')
    } else if (engineRef.current) {
      setUseLocalEngine(true)
      showToast(`Local AI ready (${localArchInfo})`, 'success')
      devDebug('Switched to local mode', { arch: localArchInfo })
    } else {
      showToast('Loading local AI...', 'info')
      devDebug('Attempting local engine init', { url: localModelUrl })
      if (await initLocalEngine()) {
        setUseLocalEngine(true)
        showToast(`Local AI ready (${localArchInfo})`, 'success')
      } else {
        showToast('Local AI failed — check model URL', 'error')
        devDebug('Local engine init failed')
      }
    }
  }, [useLocalEngine, localArchInfo, localModelUrl, showToast, initLocalEngine])

  return {
    useLocalEngine, setUseLocalEngine,
    localEngineLoading, setLocalEngineLoading,
    localArchInfo, setLocalArchInfo,
    localModelUrl, setLocalModelUrl,
    engineRef,
    engineLoadingRef,
    initLocalEngine,
    handleToggleLocalEngine,
  }
}
