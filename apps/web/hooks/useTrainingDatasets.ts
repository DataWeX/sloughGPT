'use client'

import { useState, useCallback, useRef } from 'react'
import { datasetController, filesController } from '@/lib/controllers'
import { trackEvent } from '@/lib/dev-log'
import { formatToastError } from '@/lib/error-utils'
import { sanitizeDatasetName } from '@/lib/dataset-controller'
import { ApiError } from '@/lib/http-client'
import type { Dataset, DatasetPreview } from '@/lib/dataset-controller'
import type { FileEntry } from '@/lib/files-controller'

export interface UseTrainingDatasetsReturn {
  datasets: Dataset[]
  selectedDataset: string
  loadingDatasets: boolean
  importModalOpen: boolean
  datasetPreview: DatasetPreview | null
  setSelectedDataset: (id: string) => void
  setImportModalOpen: (open: boolean) => void
  setDatasetPreview: (p: DatasetPreview | null) => void
  fetchDatasets: () => Promise<void>
}

/** Uploaded files appear in the selector as promote-on-select entries. */
function fileToDataset(f: FileEntry): Dataset {
  return {
    id: f.id,
    name: f.filename,
    source: 'my file',
    kind: 'dataset',
    size: f.size,
    created_at: f.uploaded_at,
    fromFile: true,
  }
}

/**
 * Turn an uploaded file into a trainable dataset: create the dataset (backend
 * create is idempotent — mkdir exist_ok), then append the file's text only if
 * the dataset is still empty so re-selection never duplicates rows.
 */
async function promoteFile(entry: Dataset): Promise<string> {
  const detail = await filesController.getDetail(entry.id)
  if (!detail) throw new Error('Could not read this file')
  const text = detail.text ?? ''
  if (!text.trim()) throw new Error('This file has no readable text to train on')

  const name = sanitizeDatasetName(entry.name)
  const created = await datasetController.create({ name })
  const datasetId = created?.id || name

  let empty = true
  try {
    const preview = await datasetController.preview(datasetId, 1)
    empty = !preview || preview.total_samples === 0
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 404)) throw e
  }
  if (empty) await datasetController.addData(datasetId, [text])
  return datasetId
}

export function useTrainingDatasets(
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void,
): UseTrainingDatasetsReturn {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, _setSelectedDataset] = useState('')
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [importModalOpen, setImportModalOpen] = useState(false)
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null)
  const promotingRef = useRef<Set<string>>(new Set())

  const setSelectedDataset = async (id: string) => {
    trackEvent('dataset_selected', { dataset_id: id })
    const entry = datasets.find((d) => d.fromFile && d.id === id)
    if (!entry) {
      _setSelectedDataset(id)
      return
    }
    if (promotingRef.current.has(id)) return
    promotingRef.current.add(id)
    let target: string
    try {
      target = await promoteFile(entry)
    } catch (e) {
      addToast(formatToastError(e, 'Could not use this file as a dataset'), 'error')
      return
    } finally {
      promotingRef.current.delete(id)
    }
    setDatasets((prev) => {
      const rest = prev.filter((d) => d.id !== entry.id)
      if (rest.some((d) => d.id === target)) return rest
      return [
        {
          id: target,
          name: entry.name,
          source: 'my file',
          kind: 'dataset',
          size: entry.size,
          created_at: entry.created_at,
        },
        ...rest,
      ]
    })
    _setSelectedDataset(target)
  }

  const fetchDatasets = useCallback(async () => {
    setLoadingDatasets(true)
    try {
      const [listRes, filesRes] = await Promise.allSettled([
        datasetController.list(),
        filesController?.list?.() ?? Promise.resolve([]),
      ])
      if (listRes.status === 'rejected') throw listRes.reason
      const list = listRes.value
      if (filesRes.status === 'fulfilled' && Array.isArray(filesRes.value)) {
        const taken = new Set<string>()
        for (const d of list) {
          if (d.id) taken.add(d.id)
          if (d.name) taken.add(d.name)
        }
        const fileEntries = filesRes.value
          .filter(
            (f) =>
              !taken.has(f.id) &&
              !taken.has(f.filename) &&
              !taken.has(sanitizeDatasetName(f.filename)),
          )
          .map(fileToDataset)
        setDatasets([...fileEntries, ...list])
      } else {
        setDatasets(list)
      }
    } catch (e) {
      addToast(formatToastError(e, 'Could not fetch datasets'), 'error')
    } finally {
      setLoadingDatasets(false)
    }
  }, [addToast])

  return {
    datasets,
    selectedDataset,
    loadingDatasets,
    importModalOpen,
    datasetPreview,
    setSelectedDataset,
    setImportModalOpen,
    setDatasetPreview,
    fetchDatasets,
  }
}
