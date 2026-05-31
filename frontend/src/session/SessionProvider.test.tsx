import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { useSession } from './SessionContext'
import { SessionProvider } from './SessionProvider'
import { SESSION_PAUSE_KEY, SESSION_STORAGE_KEY, useSessionStore } from './useSession'

function TestConsumer() {
  const { sessionId, hasUploads, bootstrapPaused } = useSession()
  return (
    <div>
      <span data-testid="session-id">{sessionId ?? 'none'}</span>
      <span data-testid="uploads">{String(hasUploads)}</span>
      <span data-testid="paused">{String(bootstrapPaused)}</span>
    </div>
  )
}

function renderWithProvider(ui: React.ReactElement = <TestConsumer />) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SessionProvider>{ui}</SessionProvider>
    </QueryClientProvider>,
  )
}

describe('SessionProvider', () => {
  beforeEach(() => {
    localStorage.removeItem(SESSION_STORAGE_KEY)
    localStorage.removeItem(SESSION_PAUSE_KEY)
    useSessionStore.setState({ sessionId: null, expiresAt: null, bootstrapPaused: false })
  })

  it('mints and stores a session on first mount', async () => {
    renderWithProvider()
    await waitFor(() => expect(screen.getByTestId('session-id')).toHaveTextContent('abc123demo'))
    expect(localStorage.getItem('doc-ingestion.demo.session')).toContain('abc123demo')
  })

  it('does not auto-mint while bootstrap is paused', async () => {
    localStorage.setItem(SESSION_PAUSE_KEY, '1')
    useSessionStore.setState({ sessionId: null, expiresAt: null, bootstrapPaused: true })
    renderWithProvider()
    await waitFor(() => expect(screen.getByTestId('paused')).toHaveTextContent('true'))
    expect(screen.getByTestId('session-id')).toHaveTextContent('none')
    await new Promise((resolve) => setTimeout(resolve, 400))
    expect(screen.getByTestId('session-id')).toHaveTextContent('none')
  })

  it('mints after startSession when paused', async () => {
    localStorage.setItem(SESSION_PAUSE_KEY, '1')
    useSessionStore.setState({ sessionId: null, expiresAt: null, bootstrapPaused: true })
    function Harness() {
      const { sessionId, startSession } = useSession()
      return (
        <div>
          <span data-testid="session-id">{sessionId ?? 'none'}</span>
          <button type="button" onClick={() => void startSession()}>
            Start
          </button>
        </div>
      )
    }
    renderWithProvider(<Harness />)
    expect(screen.getByTestId('session-id')).toHaveTextContent('none')
    await userEvent.click(screen.getByRole('button', { name: 'Start' }))
    await waitFor(() => expect(screen.getByTestId('session-id')).toHaveTextContent('abc123demo'))
    expect(localStorage.getItem(SESSION_PAUSE_KEY)).toBeNull()
  })
})
