const SESSION_API_BASE = 'documind_api_base_url'

export function getSessionApiBase(): string {
  try {
    return sessionStorage.getItem(SESSION_API_BASE)?.trim() ?? ''
  } catch {
    return ''
  }
}

export function setSessionApiBase(url: string): void {
  try {
    const t = url.trim()
    if (!t) sessionStorage.removeItem(SESSION_API_BASE)
    else sessionStorage.setItem(SESSION_API_BASE, t.replace(/\/$/, ''))
  } catch {
    /* ignore */
  }
}

/** Absolute or empty; no trailing slash. */
export function resolveApiRoot(): string {
  const env = import.meta.env.VITE_API_URL?.trim()
  if (env) return env.replace(/\/$/, '')
  const session = getSessionApiBase()
  if (session) return session.replace(/\/$/, '')
  if (import.meta.env.DEV) return ''
  return ''
}

/** Path like `/health` → full URL or `/api/health` in dev behind proxy. */
export function apiUrl(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`
  const root = resolveApiRoot()
  if (root) return `${root}${p}`
  if (import.meta.env.DEV) return `/api${p}`
  return p
}

export type IngestPdfResult = {
  documents_ingested: number
  chunks_created: number
  total_chunks_indexed: number
  file_name: string
  page_count: number
  tables_extracted: number
  images_embedded: number
  ocr_page_count: number
  image_ocr_count: number
}

export type RetrievedChunk = {
  chunk_id: string
  doc_id: string
  score: number
  text: string
  metadata: Record<string, unknown>
}

export type QueryResult = {
  answer: string
  cached: boolean
  route: string
  retrieved_context: RetrievedChunk[]
}

export type IndexedDocumentInfo = {
  doc_id: string
  chunk_count: number
  file_name: string | null
  source: string | null
}

export type IndexDocumentsResponse = {
  total_chunks: number
  documents: IndexedDocumentInfo[]
}

export async function fetchHealth(): Promise<{ status: string; indexed_vectors: number }> {
  const res = await fetch(apiUrl('/health'))
  if (!res.ok) throw new Error(`Health check failed (${res.status})`)
  return res.json()
}

export async function fetchIndexDocuments(): Promise<IndexDocumentsResponse> {
  const res = await fetch(apiUrl('/index/documents'))
  if (!res.ok) throw new Error(`Index list failed (${res.status})`)
  return res.json()
}

export async function clearIndex(): Promise<{ status: string; indexed_vectors: number }> {
  const res = await fetch(apiUrl('/index/clear'), { method: 'POST' })
  const text = await res.text()
  if (!res.ok) {
    let detail = text
    try {
      detail = JSON.parse(text).detail ?? text
    } catch {
      /* keep */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return JSON.parse(text) as { status: string; indexed_vectors: number }
}

export async function ingestPdf(file: File, docId?: string): Promise<IngestPdfResult> {
  const fd = new FormData()
  fd.append('file', file)
  if (docId?.trim()) fd.append('doc_id', docId.trim())
  const res = await fetch(apiUrl('/ingest/pdf'), { method: 'POST', body: fd })
  const text = await res.text()
  if (!res.ok) {
    let detail = text
    try {
      detail = JSON.parse(text).detail ?? text
    } catch {
      /* keep */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return JSON.parse(text) as IngestPdfResult
}

export async function postQuery(
  body: {
    query: string
    top_k?: number
    use_generation: boolean
    llm_base_url?: string | null
    llm_model?: string | null
    llm_route?: string | null
  },
  llmApiKey?: string | null,
): Promise<QueryResult> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (llmApiKey?.trim()) headers['X-LLM-Api-Key'] = llmApiKey.trim()
  const res = await fetch(apiUrl('/query'), {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  })
  const text = await res.text()
  if (!res.ok) {
    let detail = text
    try {
      detail = JSON.parse(text).detail ?? text
    } catch {
      /* keep */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return JSON.parse(text) as QueryResult
}
