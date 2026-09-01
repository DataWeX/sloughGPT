'use client'

import { useState, useCallback } from 'react'
import { datasetController } from '@/lib/controllers'
import { trackEvent } from '@/lib/dev-log'
import type { Dataset, DatasetPreview } from '@/lib/dataset-controller'

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

export function useTrainingDatasets(addToast: (msg: string, type?: 'success' | 'error' | 'info') => void): UseTrainingDatasetsReturn {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, _setSelectedDataset] = useState('')
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [importModalOpen, setImportModalOpen] = useState(false)
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null)

  const setSelectedDataset = (id: string) => {
    trackEvent('dataset_selected', { dataset_id: id })
    _setSelectedDataset(id)
  }

  const fetchDatasets = useCallback(async () => {
    setLoadingDatasets(true)
    try {
      const list = await datasetController.list()
      setDatasets(list)
    } catch { addToast('Could not fetch datasets', 'error') }
    finally { setLoadingDatasets(false) }
  }, [addToast])

  return {
    datasets, selectedDataset, loadingDatasets, importModalOpen, datasetPreview,
    setSelectedDataset, setImportModalOpen, setDatasetPreview, fetchDatasets,
  }
}
