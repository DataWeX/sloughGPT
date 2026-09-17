'use client'

import { filesController, type FileEntry } from '@/lib/files-controller'
import {
  datasetController,
  type Dataset,
  type ImportResponse,
  type GitHubRepo,
  type BookResult,
} from '@/lib/dataset-controller'
import { useQuery, useMutation } from '@/lib/cache'

// ─── File Hooks ─────────────────────────────────────────────────────────────

const FILE_LIST_KEY = 'files:list'

export function useFileList() {
  return useQuery(FILE_LIST_KEY, () => filesController.list(), { staleTime: 30000 })
}

export function useFileDetail(id: string | null) {
  return useQuery(['files:detail', id], () => filesController.getDetail(id!), {
    enabled: !!id,
    staleTime: 60000,
  })
}

export function useFileSearch(query: string) {
  return useQuery(['files:search', query], () => filesController.search(query), {
    enabled: query.length > 0,
    staleTime: 15000,
  })
}

export function useUploadFile() {
  return useMutation((formData: FormData) => filesController.upload(formData), {
    invalidateKeys: [FILE_LIST_KEY],
  })
}

export function useDeleteFile() {
  return useMutation((id: string) => filesController.delete(id), {
    invalidateKeys: [FILE_LIST_KEY],
  })
}

export function useDeleteFilesBatch() {
  return useMutation((ids: string[]) => filesController.deleteBatch(ids), {
    invalidateKeys: [FILE_LIST_KEY],
  })
}

export function useIngestFile() {
  return useMutation((id: string) => filesController.ingest(id), {
    invalidateKeys: [FILE_LIST_KEY],
  })
}

// ─── Dataset Hooks ──────────────────────────────────────────────────────────

const DATASET_LIST_KEY = 'datasets:list'

export function useDatasetList() {
  return useQuery(DATASET_LIST_KEY, () => datasetController.list(), { staleTime: 30000 })
}

export function useDatasetDetail(id: string | null) {
  return useQuery(['datasets:detail', id], () => datasetController.get(id!), {
    enabled: !!id,
    staleTime: 30000,
  })
}

export function useDatasetStats(id: string | null) {
  return useQuery(['datasets:stats', id], () => datasetController.getStats(id!), {
    enabled: !!id,
    staleTime: 30000,
  })
}

export function useDatasetPreview(id: string | null, limit = 10) {
  return useQuery(['datasets:preview', id, limit], () => datasetController.preview(id!, limit), {
    enabled: !!id,
    staleTime: 30000,
  })
}

export function useDeleteDataset() {
  return useMutation((id: string) => datasetController.delete(id), {
    invalidateKeys: [DATASET_LIST_KEY],
  })
}

export function useUpdateDataset() {
  return useMutation(
    ({ id, updates }: { id: string; updates: Partial<Dataset> }) =>
      datasetController.update(id, updates),
    { invalidateKeys: [DATASET_LIST_KEY] },
  )
}

// ─── Import Hooks ───────────────────────────────────────────────────────────

export function useImportFromGitHub() {
  return useMutation(
    (req: { url: string; name: string; extensions?: string[]; max_files?: number }) =>
      datasetController.importFromGitHub(req),
    { invalidateKeys: [DATASET_LIST_KEY] },
  )
}

export function useImportFromHuggingFace() {
  return useMutation(
    (req: { dataset_id: string; name?: string }) => datasetController.importFromHuggingFace(req),
    { invalidateKeys: [DATASET_LIST_KEY] },
  )
}

export function useImportFromURL() {
  return useMutation((req: { url: string; name: string }) => datasetController.importFromURL(req), {
    invalidateKeys: [DATASET_LIST_KEY],
  })
}

export function useImportFromLocal() {
  return useMutation(
    (req: { path: string; name: string; extensions?: string[] }) =>
      datasetController.importFromLocal(req),
    { invalidateKeys: [DATASET_LIST_KEY] },
  )
}

export function useImportFromKaggle() {
  return useMutation(
    (req: { dataset: string; name?: string }) => datasetController.importFromKaggle(req),
    { invalidateKeys: [DATASET_LIST_KEY] },
  )
}

export function useImportFromCSV() {
  return useMutation(
    (req: { url: string; name: string; delimiter?: string; encoding?: string }) =>
      datasetController.importFromCSV(req),
    { invalidateKeys: [DATASET_LIST_KEY] },
  )
}

export function useImportFromISBN() {
  return useMutation(
    (req: { isbn: string; name: string }) => datasetController.importFromISBN(req),
    { invalidateKeys: [DATASET_LIST_KEY] },
  )
}

// ─── Search Hooks (cached) ─────────────────────────────────────────────────

export function useSearchGitHubRepos(query: string) {
  return useQuery(['search:github', query], () => datasetController.searchGitHubRepos(query), {
    enabled: query.length > 0,
    staleTime: 60000,
  })
}

export function useSearchBooks(query: string) {
  return useQuery(['search:books', query], () => datasetController.searchBooks(query), {
    enabled: query.length > 0,
    staleTime: 60000,
  })
}

// ─── Utility Hooks ──────────────────────────────────────────────────────────

export function useConvertToMessages() {
  return useMutation(
    ({ datasetId, systemPrompt }: { datasetId: string; systemPrompt?: string }) =>
      datasetController.convertToMessages(datasetId, systemPrompt),
    { invalidateKeys: [DATASET_LIST_KEY] },
  )
}

export function useCreateDatasetVersion() {
  return useMutation((datasetId: string) => datasetController.createVersion(datasetId), {
    invalidateKeys: [DATASET_LIST_KEY],
  })
}

export function useExportDataset() {
  return useMutation(
    ({ id, format }: { id: string; format?: string }) => datasetController.export(id, format),
    {},
  )
}
