import { useEffect, useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ApiError,
  createSession,
  deleteSession,
  getSession,
  type SessionSummary,
} from '../api/client'
import { SessionContext } from './SessionContext'
import { type SessionContextValue, useSessionStore } from './useSession'

function isStaleSessionError(error: unknown) {
  return error instanceof ApiError && [400, 404, 422].includes(error.status)
}

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient()
  const sessionId = useSessionStore((s) => s.sessionId)
  const expiresAt = useSessionStore((s) => s.expiresAt)
  const bootstrapPaused = useSessionStore((s) => s.bootstrapPaused)
  const setSession = useSessionStore((s) => s.setSession)
  const clearLocalSession = useSessionStore((s) => s.clearLocalSession)
  const pauseBootstrap = useSessionStore((s) => s.pauseBootstrap)
  const resumeBootstrap = useSessionStore((s) => s.resumeBootstrap)

  const createMutation = useMutation({
    mutationFn: createSession,
    onSuccess: (session) => {
      setSession(session.session_id, session.expires_at)
      queryClient.setQueryData(['session', session.session_id], {
        ...session,
        files: [],
        total_bytes: 0,
        max_files: 3,
        max_session_bytes: 8 * 1024 * 1024,
      } satisfies SessionSummary)
    },
  })

  const mutateCreateSession = createMutation.mutate

  const sessionQuery = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => getSession(sessionId as string),
    enabled: Boolean(sessionId),
    retry: false,
    staleTime: 60_000,
    gcTime: 30 * 60_000,
  })

  useEffect(() => {
    if (bootstrapPaused) {
      return
    }
    if (sessionId || createMutation.isPending || createMutation.isError) {
      return
    }
    mutateCreateSession()
  }, [bootstrapPaused, sessionId, createMutation.isPending, createMutation.isError, mutateCreateSession])

  useEffect(() => {
    if (bootstrapPaused) {
      return
    }
    if (isStaleSessionError(sessionQuery.error)) {
      clearLocalSession()
      mutateCreateSession()
    }
  }, [bootstrapPaused, clearLocalSession, mutateCreateSession, sessionQuery.error])

  useEffect(() => {
    if (sessionQuery.data) {
      setSession(sessionQuery.data.session_id, sessionQuery.data.expires_at)
    }
  }, [sessionQuery.data, setSession])

  const isMintingSession = Boolean(!sessionId && !bootstrapPaused && createMutation.isPending)
  const awaitsSessionEnvelope =
    Boolean(sessionId) && sessionQuery.data == null && (sessionQuery.isPending || sessionQuery.isFetching)

  const value = useMemo<SessionContextValue>(
    () => ({
      sessionId,
      expiresAt,
      bootstrapPaused,
      summary: sessionQuery.data,
      isMintingSession,
      awaitsSessionEnvelope,
      isLoading: isMintingSession || awaitsSessionEnvelope,
      error: (sessionQuery.error as Error | null) ?? (createMutation.error as Error | null),
      hasUploads: Boolean(sessionQuery.data?.files.length),
      refreshSession: () => sessionQuery.refetch(),
      retrySession: async () => {
        resumeBootstrap()
        createMutation.reset()
        clearLocalSession()
        await createMutation.mutateAsync()
      },
      startSession: async () => {
        resumeBootstrap()
        createMutation.reset()
        if (!useSessionStore.getState().sessionId) {
          await createMutation.mutateAsync()
        }
      },
      logout: async () => {
        const id = useSessionStore.getState().sessionId
        pauseBootstrap()
        clearLocalSession()
        queryClient.clear()
        createMutation.reset()
        if (id) {
          try {
            await deleteSession(id)
          } catch {
            /* best-effort server cleanup */
          }
        }
      },
      clearSession: async () => {
        if (!sessionId) {
          clearLocalSession()
          await createMutation.mutateAsync()
          return
        }
        const next = await deleteSession(sessionId)
        setSession(next.session_id, null)
        queryClient.removeQueries({ queryKey: ['query'] })
        queryClient.removeQueries({ queryKey: ['session'] })
        await queryClient.invalidateQueries({ queryKey: ['session', next.session_id] })
      },
    }),
    [
      awaitsSessionEnvelope,
      bootstrapPaused,
      clearLocalSession,
      createMutation,
      expiresAt,
      isMintingSession,
      pauseBootstrap,
      queryClient,
      resumeBootstrap,
      sessionId,
      sessionQuery,
      setSession,
    ],
  )

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}
