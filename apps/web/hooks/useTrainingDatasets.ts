'use client'

import { useState, useCallback } from 'react'
import { datasetController } from '@/lib/controllers'
import type { Dataset } from '@/lib/dataset-controller'

export interface UseTrainingDatasetsReturn {
  datasets: Dataset[]
  selectedDataset: string
  loadingDatasets: boolean
  importModalOpen: boolean
  datasetPreview: { samples: Array<{ content: string }>; total_samples: number } | null
  setSelectedDataset: (id: string) => void
  setImportModalOpen: (open: boolean) => void
  setDatasetPreview: (p: { samples: Array<{ content: string }>; total_samples: number } | null) => void
  fetchDatasets: () => Promise<void>
}

export function useTrainingDatasets(addToast: (msg: string, type?: 'success' | 'error' | 'info') => void): UseTrainingDatasetsReturn {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [importModalOpen, setImportModalOpen] = useState(false)
  const [datasetPreview, setDatasetPreview] = useState<{ samples: Array<{ content: string }>; total_samples: number } | null>(null)

  const fetchDatasets = useCallback(async () => {
    setLoadingDatasets(true)
    try {
      const list = await datasetController.list()
      setDatasets(list)
    } catch { addToast('Failed to fetch datasets', 'error') }
    finally { setLoadingDatasets(false) }
  }, [addToast])

  return {
    datasets, selectedDataset, loadingDatasets, importModalOpen, datasetPreview,
    setSelectedDataset, setImportModalOpen, setDatasetPreview, fetchDatasets,
  }
}
