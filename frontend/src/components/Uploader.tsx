import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useQuery } from '@tanstack/react-query'
import { Upload } from 'lucide-react'
import { fetchRuntimeConfig, uploadDocuments, type SessionSummary, type UploadResult } from '../api/client'
import { formatBytes } from '../lib/format'

const ACCEPTED = '.pdf,.docx,.txt,.md,.html'
const MAX_FILE_BYTES = 3 * 1024 * 1024

function resultMessage(result: UploadResult) {
  const messages: Record<string, string> = {
    queued: 'Uploaded and indexed.',
    skipped: 'Duplicate upload skipped.',
    oversize: 'File exceeds the 3 MB limit.',
    file_count_cap: 'Session file count cap reached.',
    session_disk_cap: 'Session disk cap reached.',
    type_mismatch: 'File contents do not match the extension.',
  }
  return messages[result.status] ?? messages[result.message] ?? result.message
}

export function Uploader({
  sessionId,
  summary,
  onUploaded,
}: {
  sessionId: string
  summary: SessionSummary | undefined
  onUploaded: () => Promise<unknown>
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [message, setMessage] = useState('')
  const [results, setResults] = useState<UploadResult[]>([])
  const [chunkStrategyChoice, setChunkStrategyChoice] = useState<string | null>(null)
  const [embeddingProfileChoice, setEmbeddingProfileChoice] = useState<string | null>(null)
  const { data: runtimeConfig } = useQuery({
    queryKey: ['runtime-config'],
    queryFn: fetchRuntimeConfig,
    staleTime: Infinity,
  })
  const chunkStrategy = chunkStrategyChoice ?? runtimeConfig?.chunking_default_strategy ?? 'tiktoken'
  const embeddingProfile = embeddingProfileChoice ?? runtimeConfig?.embedding_default_profile ?? 'ollama_nomic'

  const mutation = useMutation({
    mutationFn: (files: File[]) => uploadDocuments(sessionId, files, { chunkStrategy, embeddingProfile }),
    onSuccess: async (response) => {
      setResults(response.results)
      setMessage('Upload finished.')
      await onUploaded()
    },
    onError: (error) => {
      setMessage(error instanceof Error ? error.message : 'Upload failed.')
    },
  })

  const upload = (fileList: FileList | File[]) => {
    if (!summary) {
      return
    }
    const files = Array.from(fileList)
    const maxFiles = summary?.max_files ?? 3
    const currentFiles = summary?.files.length ?? 0
    if (currentFiles + files.length > maxFiles) {
      setMessage(`You can upload ${Math.max(0, maxFiles - currentFiles)} more file(s).`)
      return
    }
    const oversized = files.find((file) => file.size > MAX_FILE_BYTES)
    if (oversized) {
      setMessage(`${oversized.name} is larger than ${formatBytes(MAX_FILE_BYTES)}.`)
      return
    }
    mutation.mutate(files)
  }

  return (
    <div>
      <div
        className="rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 p-8 text-center"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          upload(event.dataTransfer.files)
        }}
      >
        <Upload className="mx-auto mb-3 h-8 w-8 text-blue-600" aria-hidden="true" />
        <p className="font-medium text-slate-900">Drop files here or choose files</p>
        <p className="mt-1 text-sm text-slate-600">PDF, DOCX, TXT, Markdown, or HTML.</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          multiple
          className="sr-only"
          onChange={(event) => event.target.files && upload(event.target.files)}
        />
        <button
          type="button"
          className="mt-4 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
          disabled={mutation.isPending || !summary}
          onClick={() => inputRef.current?.click()}
        >
          {mutation.isPending ? 'Uploading...' : 'Choose files'}
        </button>
      </div>
      {runtimeConfig ? (
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <label className="block text-sm">
            <span className="mb-1 block font-medium text-slate-700">Chunking strategy</span>
            <select
              className="w-full rounded-lg border border-slate-300 bg-white p-2 text-slate-900"
              value={chunkStrategy}
              onChange={(event) => setChunkStrategyChoice(event.target.value)}
            >
              {runtimeConfig.chunking_allowed_strategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="mb-1 block font-medium text-slate-700">Embedding profile</span>
            <select
              className="w-full rounded-lg border border-slate-300 bg-white p-2 text-slate-900"
              value={embeddingProfile}
              onChange={(event) => setEmbeddingProfileChoice(event.target.value)}
            >
              {Object.keys(runtimeConfig.embedding_profiles).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}
      {message ? <p className="mt-3 text-sm text-slate-700" aria-live="polite">{message}</p> : null}
      {results.length > 0 ? (
        <ul className="mt-3 space-y-2">
          {results.map((result) => (
            <li key={`${result.filename}-${result.status}`} className="rounded-lg bg-slate-50 p-3 text-sm">
              <span className="font-medium text-slate-900">{result.filename}</span>: {resultMessage(result)}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
