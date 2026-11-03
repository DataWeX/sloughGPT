'use client'

/**
 * Stateful core of the Souls workspace.
 *
 * Owns IO and state only — the souls/checkpoints/snapshots fetches, the
 * switch/load/delete/download mutations, trait-weight editing and the
 * weight-snapshot lifecycle, plus the search filters. The tab panels and
 * dialogs own render.
 */
import { useState, useCallback, useEffect, useMemo } from 'react'
import { soulsController, type Soul, type Checkpoint } from '@/lib/souls-controller'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'
import { type Tab, type ModeInfo } from './souls-format'

export { type Tab, type ModeInfo }

export function useSoulsWorkbench() {
  const [tab, setTab] = useState<Tab>('souls')

  const [loading, setLoading] = useState(true)

  // ── Data ──
  const [souls, setSouls] = useState<Soul[]>([])
  const [currentSoul, setCurrentSoul] = useState<string | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [traitWeights, setTraitWeights] = useState<Record<string, Record<string, number>> | null>(
    null,
  )
  const [modes, setModes] = useState<Record<string, ModeInfo> | null>(null)
  const [snapshots, setSnapshots] = useState<Array<{ name: string; saved_at?: string }>>([])

  // ── UI State ──
  const [searchQuery, setSearchQuery] = useState('')
  const [switching, setSwitching] = useState<string | null>(null)
  const [loadingCheckpoint, setLoadingCheckpoint] = useState<string | null>(null)
  const [newSnapshotName, setNewSnapshotName] = useState('')
  const [editedWeights, setEditedWeights] = useState<Record<string, Record<string, number>> | null>(
    null,
  )
  const [savingWeights, setSavingWeights] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [deleteType, setDeleteType] = useState<'checkpoint' | 'snapshot'>('checkpoint')

  // ── Dialogs ──
  const [detailSoul, setDetailSoul] = useState<Soul | null>(null)
  const [compareSoul, setCompareSoul] = useState<Soul | null>(null)
  const [checkpointDetail, setCheckpointDetail] = useState<Checkpoint | null>(null)

  const addToast = useToastStore((s) => s.addToast)

  // ── Data Loading ──
  const loadData = useCallback(async () => {
    const [s, c, snaps] = await Promise.all([
      soulsController.list().catch((e) => {
        logger.warning('Could not souls list', { exception: String(e) })
        return { souls: [], current_soul: null }
      }),
      soulsController.listCheckpoints().catch((e) => {
        logger.warning('Could not checkpoints list', { exception: String(e) })
        return { checkpoints: [] }
      }),
      soulsController.listWeightSnapshots().catch((e) => {
        logger.warning('Could not weight snapshots', { exception: String(e) })
        return []
      }),
    ])
    setSouls(s.souls)
    setCurrentSoul(s.current_soul ?? null)
    setCheckpoints(c.checkpoints)
    setSnapshots(snaps)
  }, [])

  useEffect(() => {
    loadData().finally(() => setLoading(false))
  }, [loadData])

  const handleRefresh = async () => {
    await loadData()
    if (tab === 'weights') await handleLoadWeights()
    if (tab === 'snapshots') await handleLoadSnapshots()
  }

  // ── Soul Actions ──
  const handleSwitch = async (name: string) => {
    setSwitching(name)
    try {
      await soulsController.switch(name)
      setCurrentSoul(name)
      addToast(`Switched to ${name}`, 'success')
    } catch {
      addToast('Could not switch soul', 'error')
    } finally {
      setSwitching(null)
    }
  }

  // ── Checkpoint Actions ──
  const handleLoadCheckpoint = async (name: string) => {
    setLoadingCheckpoint(name)
    try {
      await soulsController.loadCheckpoint(name)
      addToast(`Loaded checkpoint: ${name}`, 'success')
      await handleRefresh()
    } catch {
      addToast('Could not load checkpoint', 'error')
    } finally {
      setLoadingCheckpoint(null)
    }
  }

  const handleDeleteCheckpoint = async (name: string) => {
    try {
      await soulsController.deleteCheckpoint(name)
      addToast(`Deleted checkpoint: ${name}`, 'success')
      setCheckpoints((prev) => prev.filter((cp) => cp.name !== name))
    } catch {
      addToast('Could not delete checkpoint', 'error')
    }
    setDeleteTarget(null)
  }

  const handleDownloadCheckpoint = async (name: string) => {
    try {
      const blob = await soulsController.downloadCheckpoint(name)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${name}.soul`
      a.click()
      URL.revokeObjectURL(url)
      addToast(`Downloaded ${name}`, 'success')
    } catch {
      addToast('Could not download', 'error')
    }
  }

  const handleCheckpointInfo = async (name: string) => {
    try {
      const info = await soulsController.checkpointInfo(name)
      if (info) setCheckpointDetail(info)
    } catch {
      addToast('Could not load checkpoint info', 'error')
    }
  }

  // ── Weight Actions ──
  const handleLoadWeights = async () => {
    try {
      const [w, m] = await Promise.all([
        soulsController.getTraitWeights(),
        soulsController.getModes(),
      ])
      setTraitWeights(w)
      setEditedWeights(JSON.parse(JSON.stringify(w)))
      setModes(m)
    } catch {
      addToast('Could not load trait weights', 'error')
    }
  }

  const handleWeightChange = (category: string, trait: string, value: number) => {
    if (!editedWeights) return
    const next = { ...editedWeights }
    next[category] = { ...next[category], [trait]: value }
    setEditedWeights(next)
  }

  const handleSaveWeights = async () => {
    if (!editedWeights) return
    setSavingWeights(true)
    try {
      await soulsController.saveTraitWeights(editedWeights)
      setTraitWeights(JSON.parse(JSON.stringify(editedWeights)))
      addToast('Trait weights saved', 'success')
    } catch {
      addToast('Could not save trait weights', 'error')
    } finally {
      setSavingWeights(false)
    }
  }

  const weightsDirty = JSON.stringify(editedWeights) !== JSON.stringify(traitWeights)

  // ── Snapshot Actions ──
  const handleLoadSnapshots = async () => {
    try {
      setSnapshots(await soulsController.listWeightSnapshots())
    } catch {
      addToast('Could not load snapshots', 'error')
    }
  }

  const handleSaveSnapshot = async () => {
    if (!newSnapshotName.trim()) return
    try {
      await soulsController.saveWeightSnapshot(newSnapshotName)
      setNewSnapshotName('')
      await handleLoadSnapshots()
      addToast('Snapshot saved', 'success')
    } catch {
      addToast('Could not save snapshot', 'error')
    }
  }

  const handleLoadSnapshot = async (name: string) => {
    try {
      await soulsController.loadWeightSnapshot(name)
      await handleLoadWeights()
      addToast('Snapshot loaded', 'success')
    } catch {
      addToast('Could not load snapshot', 'error')
    }
  }

  const handleDeleteSnapshot = async (name: string) => {
    try {
      await soulsController.deleteWeightSnapshot(name)
      await handleLoadSnapshots()
      addToast('Snapshot deleted', 'success')
    } catch {
      addToast('Could not delete snapshot', 'error')
    }
    setDeleteTarget(null)
  }

  // ── Filtered Data ──
  const filteredSouls = useMemo(
    () =>
      souls.filter(
        (s) =>
          s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.traits?.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase())) ||
          s.lineage?.toLowerCase().includes(searchQuery.toLowerCase()),
      ),
    [souls, searchQuery],
  )

  const filteredCheckpoints = useMemo(
    () =>
      checkpoints.filter(
        (cp) =>
          cp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          cp.soul?.toLowerCase().includes(searchQuery.toLowerCase()),
      ),
    [checkpoints, searchQuery],
  )

  return {
    tab,
    setTab,
    loading,
    souls,
    setSouls,
    currentSoul,
    setCurrentSoul,
    checkpoints,
    setCheckpoints,
    traitWeights,
    setTraitWeights,
    modes,
    setModes,
    snapshots,
    setSnapshots,
    searchQuery,
    setSearchQuery,
    switching,
    setSwitching,
    loadingCheckpoint,
    setLoadingCheckpoint,
    newSnapshotName,
    setNewSnapshotName,
    editedWeights,
    setEditedWeights,
    savingWeights,
    setSavingWeights,
    deleteTarget,
    setDeleteTarget,
    deleteType,
    setDeleteType,
    detailSoul,
    setDetailSoul,
    compareSoul,
    setCompareSoul,
    checkpointDetail,
    setCheckpointDetail,
    handleRefresh,
    handleSwitch,
    handleLoadCheckpoint,
    handleDeleteCheckpoint,
    handleDownloadCheckpoint,
    handleCheckpointInfo,
    handleLoadWeights,
    handleWeightChange,
    handleSaveWeights,
    weightsDirty,
    handleLoadSnapshots,
    handleSaveSnapshot,
    handleLoadSnapshot,
    handleDeleteSnapshot,
    filteredSouls,
    filteredCheckpoints,
  }
}

/** The contract the Souls panels render against. */
export type SoulsWorkbench = ReturnType<typeof useSoulsWorkbench>
