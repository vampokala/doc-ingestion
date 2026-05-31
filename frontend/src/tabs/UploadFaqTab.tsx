import { useQuery } from '@tanstack/react-query'
import { fetchRuntimeConfig } from '../api/client'

const CHUNKING_DESCRIPTIONS: Record<string, string> = {
  tiktoken: 'Best default for mixed/general documents and stable token-length chunks.',
  spacy: 'Sentence-aware chunking; good when semantic sentence boundaries matter.',
  nltk: 'Lightweight sentence tokenization; useful fallback if spaCy models are unavailable.',
  medical: 'Domain-oriented segmentation (clinical headings and terminology patterns).',
  legal: 'Domain-oriented segmentation (clauses, sections, legal citation patterns).',
}

export function UploadFaqTab() {
  const { data: runtimeConfig, isLoading, isError } = useQuery({
    queryKey: ['runtime-config'],
    queryFn: fetchRuntimeConfig,
    staleTime: Infinity,
  })

  return (
    <div className="space-y-5">
      <section className="app-card p-5">
        <h2 className="text-lg font-semibold text-slate-950">Upload FAQ</h2>
        <p className="mt-1 text-sm text-slate-600">
          Use this guide to choose chunking strategy and embedding profile before uploading. Options below reflect what
          this API server exposes via{' '}
          <code className="rounded bg-slate-100 px-1 font-mono text-xs">GET /config/runtime</code>.
        </p>
      </section>

      {isLoading ? (
        <section className="app-card p-5">
          <p className="text-sm text-slate-600">Loading configuration…</p>
        </section>
      ) : null}

      {isError ? (
        <section className="app-card border border-amber-200 bg-amber-50 p-5">
          <p className="text-sm font-medium text-amber-900">Could not load runtime configuration.</p>
          <p className="mt-1 text-sm text-amber-900">
            Start the API and refresh this page. Chunk and embedding choices in the uploader still come from the server;
            this FAQ needs the same endpoint.
          </p>
        </section>
      ) : null}

      {runtimeConfig ? (
        <>
          <section className="app-card space-y-3 p-5">
            <h3 className="text-base font-semibold text-slate-900">Chunking strategy: which one should I pick?</h3>
            <p className="text-sm text-slate-600">
              Default on this server:{' '}
              <strong className="font-semibold text-slate-800">{runtimeConfig.chunking_default_strategy}</strong>.
            </p>
            <ul className="space-y-3 text-sm text-slate-700">
              {runtimeConfig.chunking_allowed_strategies.map((strategy) => (
                <li key={strategy}>
                  <strong className="font-semibold text-slate-900">{strategy}</strong>
                  :{' '}
                  {CHUNKING_DESCRIPTIONS[strategy] ?? (
                    <span className="text-slate-600">See server documentation for this strategy.</span>
                  )}
                </li>
              ))}
            </ul>
          </section>

          <section className="app-card space-y-3 p-5">
            <h3 className="text-base font-semibold text-slate-900">Embedding profile: how to choose?</h3>
            <p className="text-sm text-slate-600">
              Default profile:{' '}
              <strong className="font-semibold text-slate-800">{runtimeConfig.embedding_default_profile}</strong>. Keep
              the same embedding profile during upload and on the Query tab for consistent retrieval against session
              uploads.
            </p>
            <dl className="space-y-4 text-sm text-slate-700">
              {Object.entries(runtimeConfig.embedding_profiles).map(([name, profile]) => (
                <div key={name} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <dt className="font-semibold text-slate-900">{name}</dt>
                  <dd className="mt-2 grid gap-1 text-slate-700">
                    <span>
                      <span className="text-slate-500">Provider:</span> {String(profile.provider ?? '—')}
                    </span>
                    <span>
                      <span className="text-slate-500">Framework:</span> {String(profile.framework ?? '—')}
                    </span>
                    <span>
                      <span className="text-slate-500">Model:</span>{' '}
                      <code className="rounded bg-white px-1 font-mono text-xs">{String(profile.model ?? '—')}</code>
                    </span>
                    <span>
                      <span className="text-slate-500">Dimension:</span> {String(profile.dimension ?? '—')}
                    </span>
                  </dd>
                </div>
              ))}
            </dl>
          </section>

          <section className="app-card space-y-3 p-5">
            <h3 className="text-base font-semibold text-slate-900">Recommended quick presets</h3>
            <ul className="space-y-2 text-sm text-slate-700">
              <li>
                <strong className="font-semibold text-slate-900">General docs:</strong>{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">tiktoken</code> +{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">
                  {runtimeConfig.embedding_default_profile}
                </code>{' '}
                (server defaults).
              </li>
              <li>
                <strong className="font-semibold text-slate-900">Higher-quality local vectors:</strong>{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">tiktoken</code> +{' '}
                {runtimeConfig.embedding_profiles.st_mpnet_base ? (
                  <code className="rounded bg-slate-100 px-1 font-mono text-xs">st_mpnet_base</code>
                ) : (
                  <code className="rounded bg-slate-100 px-1 font-mono text-xs">st_minilm</code>
                )}{' '}
                (compare latency vs quality on your hardware).
              </li>
              <li>
                <strong className="font-semibold text-slate-900">Medical notes:</strong>{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">medical</code> + a{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">sentence_transformers</code> profile such as{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">st_minilm</code> or{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">st_mpnet_base</code>.
              </li>
              <li>
                <strong className="font-semibold text-slate-900">Legal docs:</strong>{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">legal</code> +{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">st_minilm</code> or{' '}
                <code className="rounded bg-slate-100 px-1 font-mono text-xs">st_mpnet_base</code>.
              </li>
            </ul>
          </section>
        </>
      ) : null}
    </div>
  )
}
