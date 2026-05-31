import { create } from 'zustand'
import type { SessionSummary } from '../api/client'

export const SESSION_STORAGE_KEY = 'doc-ingestion.demo.session'
/** When set, auto session mint is skipped until `resumeBootstrap()` / explicit start. */
export const SESSION_PAUSE_KEY = 'doc-ingestion.demo.session-paused'

function readPaused(): boolean {
  try {
    return localStorage.getItem(SESSION_PAUSE_KEY) === '1'
  } catch {
    return false
  }
}

function readStoredSessionBody(): Pick<SessionState, 'sessionId' | 'expiresAt'> {
  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY)
    if (!raw) {
      return { sessionId: null, expiresAt: null }
    }
    const parsed = JSON.parse(raw) as { sessionId?: string; expiresAt?: number }
    return { sessionId: parsed.sessionId ?? null, expiresAt: parsed.expiresAt ?? null }
  } catch {
    return { sessionId: null, expiresAt: null }
  }
}

function readInitialSessionFields(): Pick<SessionState, 'sessionId' | 'expiresAt' | 'bootstrapPaused'> {
  if (readPaused()) {
    return { sessionId: null, expiresAt: null, bootstrapPaused: true }
  }
  return { ...readStoredSessionBody(), bootstrapPaused: false }
}

interface SessionState {
  sessionId: string | null
  expiresAt: number | null
  bootstrapPaused: boolean
  setSession: (sessionId: string, expiresAt: number | null) => void
  clearLocalSession: () => void
  pauseBootstrap: () => void
  resumeBootstrap: () => void
}

export const useSessionStore = create<SessionState>((set) => ({
  ...readInitialSessionFields(),
  setSession: (sessionId, expiresAt) => {
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({ sessionId, expiresAt }))
    set({ sessionId, expiresAt })
  },
  clearLocalSession: () => {
    localStorage.removeItem(SESSION_STORAGE_KEY)
    set({ sessionId: null, expiresAt: null })
  },
  pauseBootstrap: () => {
    localStorage.setItem(SESSION_PAUSE_KEY, '1')
    set({ bootstrapPaused: true })
  },
  resumeBootstrap: () => {
    localStorage.removeItem(SESSION_PAUSE_KEY)
    set({ bootstrapPaused: false })
  },
}))

export interface SessionContextValue {
  sessionId: string | null
  expiresAt: number | null
  bootstrapPaused: boolean
  summary: SessionSummary | undefined
  /** True only while POST /sessions is in-flight (nothing in localStorage yet). */
  isMintingSession: boolean
  /**
   * True while we lack session summary yet (new session bootstrap or hydrating from storage).
   * Prefer this or isMintingSession over isLoading when you care about UX flicker.
   */
  awaitsSessionEnvelope: boolean
  isLoading: boolean
  error: Error | null
  hasUploads: boolean
  refreshSession: () => Promise<unknown>
  clearSession: () => Promise<void>
  logout: () => Promise<void>
  startSession: () => Promise<void>
  retrySession: () => Promise<void>
}
