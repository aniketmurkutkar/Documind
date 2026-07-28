import { useCallback, useEffect, useState } from 'react'
import {
  apiUrl,
  clearIndex,
  fetchHealth,
  fetchIndexDocuments,
  getSessionApiBase,
  ingestPdf,
  postQuery,
  setSessionApiBase,
  type IndexDocumentsResponse,
  type IngestPdfResult,
  type QueryResult,
} from './api'
import './App.css'

const SK = {
  llmKey: 'documind_llm_api_key',
  llmBase: 'documind_llm_base_url',
  llmModel: 'documind_llm_model',
  llmRoute: 'documind_llm_route',
} as const

function readSession(key: string): string {
  try {
    return sessionStorage.getItem(key) ?? ''
  } catch {
    return ''
  }
}

function writeSession(key: string, value: string): void {
  try {
    const v = value.trim()
    if (!v) sessionStorage.removeItem(key)
    else sessionStorage.setItem(key, v)
  } catch {
    /* ignore */
  }
}

export default function App() {
  const [health, setHealth] = useState<{ ok: boolean; vectors: number } | null>(null)
  const [corpus, setCorpus] = useState<IndexDocumentsResponse | null>(null)
  const [healthErr, setHealthErr] = useState<string | null>(null)
  const [clearBusy, setClearBusy] = useState(false)

  const [apiBaseInput, setApiBaseInput] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(true)

  const [llmKey, setLlmKey] = useState('')
  const [llmBase, setLlmBase] = useState('https://api.openai.com/v1')
  const [llmModel, setLlmModel] = useState('gpt-4o-mini')
  const [llmRoute, setLlmRoute] = useState<'chat_completions' | 'responses'>(
    'chat_completions',
  )

  const [ingestBusy, setIngestBusy] = useState(false)
  const [ingestResult, setIngestResult] = useState<IngestPdfResult | null>(null)
  const [ingestErr, setIngestErr] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(5)
  const [useGen, setUseGen] = useState(true)
  const [queryBusy, setQueryBusy] = useState(false)
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [queryErr, setQueryErr] = useState<string | null>(null)
  const [sourcesOpen, setSourcesOpen] = useState(true)

  const refreshStatus = useCallback(async () => {
    setHealthErr(null)
    try {
      const [h, idx] = await Promise.all([fetchHealth(), fetchIndexDocuments()])
      setHealth({ ok: h.status === 'ok', vectors: h.indexed_vectors })
      setCorpus(idx)
    } catch (e) {
      setHealth(null)
      setCorpus(null)
      setHealthErr(e instanceof Error ? e.message : 'Could not reach API')
    }
  }, [])

  useEffect(() => {
    setApiBaseInput(getSessionApiBase())
    setLlmKey(readSession(SK.llmKey))
    setLlmBase(readSession(SK.llmBase) || 'https://api.openai.com/v1')
    setLlmModel(readSession(SK.llmModel) || 'gpt-4o-mini')
    const r = readSession(SK.llmRoute)
    setLlmRoute(r === 'responses' ? 'responses' : 'chat_completions')
  }, [])

  useEffect(() => {
    void refreshStatus()
  }, [refreshStatus])

  const saveSettings = () => {
    setSessionApiBase(apiBaseInput)
    writeSession(SK.llmKey, llmKey)
    writeSession(SK.llmBase, llmBase)
    writeSession(SK.llmModel, llmModel)
    writeSession(SK.llmRoute, llmRoute)
    void refreshStatus()
  }

  const clearLlmKey = () => {
    setLlmKey('')
    writeSession(SK.llmKey, '')
  }

  const onPdf = async (file: File | null) => {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      setIngestErr('Choose a PDF file.')
      return
    }
    setIngestBusy(true)
    setIngestErr(null)
    setIngestResult(null)
    try {
      const r = await ingestPdf(file)
      setIngestResult(r)
      await refreshStatus()
    } catch (e) {
      setIngestErr(e instanceof Error ? e.message : 'Ingest failed')
    } finally {
      setIngestBusy(false)
    }
  }

  const runQuery = async () => {
    setQueryBusy(true)
    setQueryErr(null)
    setQueryResult(null)
    try {
      const r = await postQuery(
        {
          query: question.trim(),
          top_k: topK,
          use_generation: useGen,
          llm_base_url: llmBase.trim() || null,
          llm_model: llmModel.trim() || null,
          llm_route: llmRoute,
        },
        llmKey.trim() || null,
      )
      setQueryResult(r)
    } catch (e) {
      setQueryErr(e instanceof Error ? e.message : 'Query failed')
    } finally {
      setQueryBusy(false)
    }
  }

  const onClearIndex = async () => {
    if (
      !window.confirm(
        'Remove all documents from the index? This clears every PDF in memory for this API process.',
      )
    ) {
      return
    }
    setClearBusy(true)
    setIngestErr(null)
    try {
      await clearIndex()
      setIngestResult(null)
      await refreshStatus()
    } catch (e) {
      setIngestErr(e instanceof Error ? e.message : 'Clear index failed')
    } finally {
      setClearBusy(false)
    }
  }

  const devHint = import.meta.env.DEV ? `Proxy: ${apiUrl('/health')}` : null

  return (
    <div className="layout">
      <header className="top">
        <div>
          <h1 className="title">Documind</h1>
          <p className="tagline">
            Multimodal RAG — upload a PDF, ask questions, get grounded answers.
          </p>
        </div>
        <div className="health">
          {health && (
            <span className="pill ok">
              API · {health.vectors} chunks
              {corpus !== null && corpus.documents.length > 0 && (
                <> · {corpus.documents.length} doc{corpus.documents.length === 1 ? '' : 's'}</>
              )}
            </span>
          )}
          {healthErr && <span className="pill err">{healthErr}</span>}
          <button type="button" className="btn ghost sm" onClick={() => void refreshStatus()}>
            Refresh
          </button>
        </div>
      </header>

      <section className="card">
        <button
          type="button"
          className="card-head"
          onClick={() => setSettingsOpen((o) => !o)}
          aria-expanded={settingsOpen}
        >
          <span>Settings · API &amp; your key (BYOK)</span>
          <span className="chev">{settingsOpen ? '▼' : '▶'}</span>
        </button>
        {settingsOpen && (
          <div className="card-body stack">
            <p className="hint">
              Keys stay in this browser tab (<code>sessionStorage</code>) and are sent to{' '}
              <strong>your</strong> Documind backend only — never committed to GitHub. Use HTTPS
              when deployed.
            </p>
            <label className="field">
              <span>Backend base URL (optional)</span>
              <input
                className="input"
                placeholder="empty = dev proxy (/api → localhost:8000)"
                value={apiBaseInput}
                onChange={(e) => setApiBaseInput(e.target.value)}
              />
            </label>
            <label className="field">
              <span>LLM API key (optional)</span>
              <input
                className="input mono"
                type="password"
                autoComplete="off"
                placeholder="sk-… or your provider key"
                value={llmKey}
                onChange={(e) => setLlmKey(e.target.value)}
              />
            </label>
            <div className="row2">
              <label className="field">
                <span>LLM base URL</span>
                <input
                  className="input mono"
                  value={llmBase}
                  onChange={(e) => setLlmBase(e.target.value)}
                />
              </label>
              <label className="field">
                <span>Model</span>
                <input
                  className="input mono"
                  value={llmModel}
                  onChange={(e) => setLlmModel(e.target.value)}
                />
              </label>
            </div>
            <label className="field">
              <span>Route</span>
              <select
                className="input"
                value={llmRoute}
                onChange={(e) =>
                  setLlmRoute(e.target.value === 'responses' ? 'responses' : 'chat_completions')
                }
              >
                <option value="chat_completions">chat_completions (OpenAI-style)</option>
                <option value="responses">responses (e.g. Groq Responses API)</option>
              </select>
            </label>
            <div className="actions">
              <button type="button" className="btn primary" onClick={saveSettings}>
                Save to browser session
              </button>
              <button type="button" className="btn ghost" onClick={clearLlmKey}>
                Clear key
              </button>
            </div>
            {devHint && <p className="mono subtle">{devHint}</p>}
          </div>
        )}
      </section>

      <section className="card">
        <h2 className="h2">1 · Ingest PDF</h2>
        <p className="sub">
          Text and tables are extracted locally; embeddings use the server model (no LLM tokens
          for this step).
        </p>
        <div
          className={`drop ${dragActive ? 'active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragActive(true)
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragActive(false)
            const f = e.dataTransfer.files[0]
            void onPdf(f ?? null)
          }}
        >
          <input
            type="file"
            accept="application/pdf,.pdf"
            className="file-input"
            id="pdf"
            onChange={(e) => void onPdf(e.target.files?.[0] ?? null)}
          />
          <label htmlFor="pdf" className="drop-label">
            {ingestBusy ? 'Indexing…' : 'Drop a PDF here or click to choose'}
          </label>
        </div>
        <div className="ingest-actions">
          <button
            type="button"
            className="btn ghost danger"
            disabled={clearBusy || !corpus?.total_chunks}
            onClick={() => void onClearIndex()}
          >
            {clearBusy ? 'Clearing…' : 'Clear entire index'}
          </button>
        </div>
        {ingestErr && <p className="err">{ingestErr}</p>}
        {corpus && (
          <div className="corpus-panel">
            <h3 className="h3">Documents in memory</h3>
            {corpus.documents.length === 0 ? (
              <p className="sub flush">No documents indexed yet.</p>
            ) : (
              <ul className="corpus-list">
                {corpus.documents.map((d) => (
                  <li key={d.doc_id} className="corpus-item">
                    <div className="corpus-title">
                      {d.file_name ?? d.doc_id}
                      {d.source && <span className="corpus-badge">{d.source}</span>}
                    </div>
                    <div className="corpus-meta">
                      <code>{d.doc_id}</code>
                      <span>· {d.chunk_count} chunks</span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
            <p className="sub flush">
              Total chunks in index: <strong>{corpus.total_chunks}</strong>
            </p>
          </div>
        )}
        {ingestResult && (
          <div className="result">
            <p className="result-label">Last ingest</p>
            <p>
              <strong>{ingestResult.file_name}</strong> — {ingestResult.page_count} pages,{' '}
              {ingestResult.chunks_created} new chunks (index total{' '}
              {ingestResult.total_chunks_indexed}).
            </p>
            <p className="sub">
              Tables: {ingestResult.tables_extracted} · Images: {ingestResult.images_embedded} ·
              OCR pages: {ingestResult.ocr_page_count}
            </p>
          </div>
        )}
      </section>

      <section className="card">
        <h2 className="h2">2 · Ask a question</h2>
        <label className="field">
          <span>Question</span>
          <textarea
            className="input area"
            rows={3}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What does the document say about…?"
          />
        </label>
        <div className="row2">
          <label className="field">
            <span>Top K retrieval</span>
            <input
              className="input"
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 5)}
            />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={useGen}
              onChange={(e) => setUseGen(e.target.checked)}
            />
            Generate answer with LLM
          </label>
        </div>
        <button
          type="button"
          className="btn primary"
          disabled={queryBusy || question.trim().length < 2}
          onClick={() => void runQuery()}
        >
          {queryBusy ? 'Running…' : 'Run query'}
        </button>
        {queryErr && <p className="err">{queryErr}</p>}
        {queryResult && (
          <div className="answer-block">
            <div className="meta">
              <span className="pill dim">route: {queryResult.route}</span>
              {queryResult.cached && <span className="pill dim">cached</span>}
            </div>
            <div className="answer">{queryResult.answer}</div>
            <button
              type="button"
              className="card-head sources-toggle"
              onClick={() => setSourcesOpen((o) => !o)}
            >
              <span>Retrieved chunks ({queryResult.retrieved_context.length})</span>
              <span className="chev">{sourcesOpen ? '▼' : '▶'}</span>
            </button>
            {sourcesOpen && (
              <ul className="sources">
                {queryResult.retrieved_context.map((c) => (
                  <li key={c.chunk_id}>
                    <div className="src-head">
                      <code>{c.doc_id}</code>
                      <span className="score">score {c.score.toFixed(3)}</span>
                    </div>
                    <pre className="src-text">{c.text}</pre>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </section>

      <footer className="foot">
        OpenAPI docs: <code>{apiUrl('/docs')}</code> (when backend is reachable)
      </footer>
    </div>
  )
}
