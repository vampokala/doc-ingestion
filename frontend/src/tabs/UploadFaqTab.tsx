export function UploadFaqTab() {
  return (
    <div className="space-y-5">
      <section className="app-card p-5">
        <h2 className="text-lg font-semibold text-slate-950">Upload FAQ</h2>
        <p className="mt-1 text-sm text-slate-600">
          Use this guide to choose the best chunking strategy and embedding profile before uploading.
        </p>
      </section>

      <section className="app-card space-y-3 p-5">
        <h3 className="text-base font-semibold text-slate-900">Chunking strategy: which one should I pick?</h3>
        <ul className="space-y-2 text-sm text-slate-700">
          <li><strong>tiktoken</strong>: best default for mixed/general documents and stable token-length chunks.</li>
          <li><strong>spacy</strong>: sentence-aware chunking; good when semantic sentence boundaries matter.</li>
          <li><strong>nltk</strong>: lightweight sentence tokenization; useful fallback if spaCy models are unavailable.</li>
          <li><strong>medical</strong>: domain-oriented segmentation (clinical headings and terminology patterns).</li>
          <li><strong>legal</strong>: domain-oriented segmentation (clauses, sections, legal citation patterns).</li>
        </ul>
      </section>

      <section className="app-card space-y-3 p-5">
        <h3 className="text-base font-semibold text-slate-900">Embedding profile: how to choose?</h3>
        <ul className="space-y-2 text-sm text-slate-700">
          <li>
            <strong>ollama_nomic</strong>: local embedding via Ollama (`nomic-embed-text`); good local-first default.
          </li>
          <li>
            <strong>st_minilm</strong>: sentence-transformers (`all-MiniLM-L6-v2`); typically faster and lower dimension.
          </li>
        </ul>
        <p className="text-sm text-slate-700">
          Keep the same embedding profile during upload and query for consistent retrieval results.
        </p>
      </section>

      <section className="app-card space-y-3 p-5">
        <h3 className="text-base font-semibold text-slate-900">Recommended quick presets</h3>
        <ul className="space-y-2 text-sm text-slate-700">
          <li><strong>General docs:</strong> `tiktoken` + `ollama_nomic`</li>
          <li><strong>Medical notes:</strong> `medical` + `st_minilm` (or compare against `ollama_nomic`)</li>
          <li><strong>Legal docs:</strong> `legal` + `st_minilm` (or compare against `ollama_nomic`)</li>
        </ul>
      </section>
    </div>
  )
}
