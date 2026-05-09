import * as Tabs from '@radix-ui/react-tabs'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AlertCircle, BookOpen, CircleHelp, Database, FileText, Fingerprint, LogOut } from 'lucide-react'
import { useMemo, useState } from 'react'
import { QueryTab } from './tabs/QueryTab'
import { OverviewTab } from './tabs/OverviewTab'
import { DocumentsTab } from './tabs/DocumentsTab'
import { UploadFaqTab } from './tabs/UploadFaqTab'
import { SessionProvider } from './session/SessionProvider'
import { useSession } from './session/SessionContext'
import { formatTtl } from './lib/format'

const NEW_SESSION_HINT =
  'Fresh ID for uploads in this browser. Replaces any current demo session (including uploads on the server).'

function SessionErrorBanner({ message, onRetry }: { message: string; onRetry: () => void }) {
  const collapsedSummary =
    message.includes('DOC_PROFILE=demo') || message.includes('Demo sessions are disabled')
      ? 'Demo sessions are disabled on this server.'
      : null
  return (
    <div className="mt-3 flex flex-wrap items-start justify-between gap-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
      <div className="min-w-0 flex-1">
        <span className="inline-flex items-start gap-2">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <span className="min-w-0">
            {collapsedSummary ? (
              <>
                <span className="font-semibold">{collapsedSummary}</span>
                <details className="mt-1">
                  <summary className="cursor-pointer font-semibold underline">Details</summary>
                  <p className="mt-1 max-h-32 overflow-y-auto whitespace-pre-wrap break-words">{message}</p>
                </details>
              </>
            ) : (
              <span className="max-h-32 overflow-y-auto whitespace-pre-wrap break-words">{message}</span>
            )}
          </span>
        </span>
      </div>
      <button type="button" className="shrink-0 font-semibold underline" onClick={() => void onRetry()}>
        Retry session
      </button>
    </div>
  )
}

function Shell() {
  const {
    sessionId,
    expiresAt,
    error,
    retrySession,
    isMintingSession,
    awaitsSessionEnvelope,
    isLoading,
    clearSession,
    logout,
    startSession,
    bootstrapPaused,
  } = useSession()
  const [activeTab, setActiveTab] = useState('overview')

  const sessionStatusLabel = (() => {
    if (bootstrapPaused) {
      return null
    }
    if (isMintingSession) {
      return 'Starting session…'
    }
    if (awaitsSessionEnvelope) {
      return 'Loading session…'
    }
    if (sessionId) {
      return `Session …${sessionId.slice(-5)}`
    }
    return 'Starting session…'
  })()

  return (
    <main className="flex min-h-screen flex-col bg-slate-100 px-4 py-6 md:px-8">
      <div className="mx-auto flex min-h-0 w-full max-w-6xl flex-1 flex-col gap-5">
        <header className="app-card shrink-0 p-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-blue-700">Doc Ingestion</p>
              <h1 className="mt-1 text-3xl font-bold text-slate-950">Document Q&A Assistant</h1>
              <p className="mt-2 max-w-3xl text-slate-600">
                Ask citation-aware questions against the global demo corpus, your private uploads, or both.
              </p>
            </div>

            {bootstrapPaused ? (
              <div className="flex max-w-md flex-col gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                <p className="font-medium text-slate-800">Not signed in to a demo session</p>
                <p className="text-xs text-slate-600">
                  Start a session to upload documents in this browser, or stay logged out to browse Overview and Query
                  against the global corpus only.
                </p>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-blue-700"
                    onClick={() => void startSession()}
                  >
                    <Fingerprint className="h-4 w-4 shrink-0" aria-hidden="true" />
                    Start session
                  </button>
                  <button
                    type="button"
                    className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
                    onClick={() => setActiveTab('upload-faq')}
                  >
                    <CircleHelp className="h-4 w-4 shrink-0" aria-hidden="true" />
                    Upload FAQ
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700 md:items-end">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 md:justify-end">
                  {sessionStatusLabel ? <span className="font-medium text-slate-800">{sessionStatusLabel}</span> : null}
                  {expiresAt ? (
                    <span className="text-slate-500">TTL {formatTtl(expiresAt)}</span>
                  ) : !bootstrapPaused && sessionId ? (
                    <span className="text-slate-500">TTL —</span>
                  ) : null}
                </div>
                <div className="flex flex-wrap gap-2 md:justify-end">
                  <button
                    type="button"
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-800 hover:bg-slate-100 disabled:pointer-events-none disabled:opacity-50"
                    disabled={isLoading}
                    title={NEW_SESSION_HINT}
                    aria-busy={isLoading}
                    onClick={() => void clearSession()}
                  >
                    <Fingerprint className="h-4 w-4 shrink-0" aria-hidden="true" />
                    {isLoading ? 'Working…' : 'New session'}
                  </button>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-800 hover:bg-slate-100 disabled:pointer-events-none disabled:opacity-50"
                    disabled={isLoading}
                    title="Clear demo session from this browser and pause automatic session creation."
                    onClick={() => void logout()}
                  >
                    <LogOut className="h-4 w-4 shrink-0" aria-hidden="true" />
                    Logout
                  </button>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm font-semibold text-blue-700 underline hover:text-blue-900"
                    onClick={() => setActiveTab('upload-faq')}
                  >
                    Upload FAQ
                  </button>
                </div>
              </div>
            )}
          </div>

          {error ? <SessionErrorBanner message={error.message} onRetry={retrySession} /> : null}
        </header>

        <Tabs.Root value={activeTab} onValueChange={setActiveTab} className="flex min-h-0 flex-1 flex-col gap-5">
          <Tabs.List className="app-card inline-flex shrink-0 flex-wrap gap-2 p-2" aria-label="Main sections">
            <Tabs.Trigger
              value="overview"
              className="inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold text-slate-700 data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              <BookOpen className="h-4 w-4" aria-hidden="true" />
              Overview
            </Tabs.Trigger>
            <Tabs.Trigger
              value="query"
              className="inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold text-slate-700 data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              <Database className="h-4 w-4" aria-hidden="true" />
              Query
            </Tabs.Trigger>
            <Tabs.Trigger
              value="documents"
              className="inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold text-slate-700 data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              <FileText className="h-4 w-4" aria-hidden="true" />
              My documents
            </Tabs.Trigger>
            <Tabs.Trigger
              value="upload-faq"
              className="inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold text-slate-700 data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              <CircleHelp className="h-4 w-4" aria-hidden="true" />
              Upload FAQ
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="overview" className="min-h-0 flex-1 overflow-y-auto outline-none">
            <OverviewTab />
          </Tabs.Content>
          <Tabs.Content value="query" className="min-h-0 flex-1 overflow-y-auto outline-none">
            <QueryTab />
          </Tabs.Content>
          <Tabs.Content value="documents" className="flex min-h-0 flex-1 flex-col overflow-hidden outline-none">
            <DocumentsTab onOpenUploadFaq={() => setActiveTab('upload-faq')} />
          </Tabs.Content>
          <Tabs.Content value="upload-faq" className="min-h-0 flex-1 overflow-y-auto outline-none">
            <UploadFaqTab />
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </main>
  )
}

function App() {
  const queryClient = useMemo(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            retry: false,
          },
        },
      }),
    [],
  )

  return (
    <QueryClientProvider client={queryClient}>
      <SessionProvider>
        <Shell />
      </SessionProvider>
    </QueryClientProvider>
  )
}

export default App
