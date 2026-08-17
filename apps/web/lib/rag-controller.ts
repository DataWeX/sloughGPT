/**
 * RAG Controller — Frontend API client for the production RAG system.
 *
 * Manages document ingestion, querying, verification, and index stats
 * via the `/knowledge/rag/*` backend endpoints.
 */

import { apiGet, apiPost } from './http-client'

export interface RAGDocument {
  metadata: Record<string, unknown>
  chunk_size: number
  num_chunks: number
  added_at: number
}

export interface RAGStats {
  total_documents: number
  total_chunks: number
  index_size: number
}

export interface RAGQueryResult {
  chunk_id: string
  content: string
  score: number
  rank: number
  metadata: Record<string, unknown>
}

export interface RAGQueryResponse {
  question: string
  results: RAGQueryResult[]
  context: string
  num_results: number
}

export interface RAGIngestResponse {
  chunk_ids: string[]
  num_chunks: number
  stats: RAGStats
}

export interface RAGVerifyResponse {
  original_text: string
  question: string
  verification: {
    total_claims: number
    grounded_claims: Array<{
      subject: string
      predicate: string
      confidence: number
      sources: string[]
    }>
    hallucinations: Array<{
      subject: string
      predicate: string
      reason: string
      confidence: number
    }>
    overall_confidence: number
    hallucination_rate: number
    formatted_citations: string
  }
  citations: string
  confidence: number
  is_verified: boolean
}

/**
 * Ingest a document into the RAG index.
 */
export async function ingestDocument(
  content: string,
  source: string = 'user',
  topic: string = 'general',
  chunkSize: number = 512,
): Promise<RAGIngestResponse> {
  return apiPost('/knowledge/rag/ingest', {
    content,
    source,
    topic,
    chunk_size: chunkSize,
  }) as Promise<RAGIngestResponse>
}

/**
 * Query the RAG index for relevant context.
 */
export async function queryRAG(
  question: string,
  topK: number = 5,
): Promise<RAGQueryResponse> {
  return apiPost('/knowledge/rag/query', { question, top_k: topK }) as Promise<RAGQueryResponse>
}

/**
 * Verify generated text against the RAG index.
 */
export async function verifyRAG(
  text: string,
  question: string,
): Promise<RAGVerifyResponse> {
  return apiPost('/knowledge/rag/verify', { text, question }) as Promise<RAGVerifyResponse>
}

/**
 * List all documents in the RAG index.
 */
export async function listRAGDocuments(): Promise<{
  documents: RAGDocument[]
  stats: RAGStats
}> {
  return apiGet('/knowledge/rag/documents') as Promise<{ documents: RAGDocument[]; stats: RAGStats }>
}

/**
 * Get RAG index statistics.
 */
export async function getRAGStats(): Promise<RAGStats> {
  return apiGet('/knowledge/rag/stats') as Promise<RAGStats>
}

/**
 * Clear the entire RAG index.
 */
export async function clearRAG(): Promise<{ cleared: number }> {
  return apiPost('/knowledge/rag/clear') as Promise<{ cleared: number }>
}
