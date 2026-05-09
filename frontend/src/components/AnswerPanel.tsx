import { Check, Copy } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { QueryResponseModel } from '../api/generated'

const markdownComponents = {
  h1: ({ ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 className="mb-3 mt-4 text-xl font-bold text-slate-950 first:mt-0" {...props} />
  ),
  h2: ({ ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 className="mb-2 mt-4 text-lg font-semibold text-slate-950 first:mt-0" {...props} />
  ),
  h3: ({ ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3 className="mb-2 mt-3 text-base font-semibold text-slate-900 first:mt-0" {...props} />
  ),
  p: ({ ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className="mb-3 leading-relaxed last:mb-0" {...props} />
  ),
  ul: ({ ...props }: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className="mb-3 list-disc space-y-1 pl-5" {...props} />
  ),
  ol: ({ ...props }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className="mb-3 list-decimal space-y-1 pl-5" {...props} />
  ),
  li: ({ ...props }: React.HTMLAttributes<HTMLLIElement>) => <li className="leading-relaxed" {...props} />,
  strong: ({ ...props }: React.HTMLAttributes<HTMLElement>) => (
    <strong className="font-semibold text-slate-950" {...props} />
  ),
  code: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => {
    const isBlock = typeof className === 'string' && className.includes('language-')
    if (isBlock) {
      return (
        <code className={`block overflow-x-auto rounded-lg bg-slate-900 p-3 text-sm text-slate-100 ${className ?? ''}`} {...props}>
          {children}
        </code>
      )
    }
    return (
      <code className="rounded bg-slate-200 px-1 py-0.5 font-mono text-[0.9em] text-slate-900" {...props}>
        {children}
      </code>
    )
  },
  pre: ({ children, ...props }: React.HTMLAttributes<HTMLPreElement>) => (
    <pre className="mb-3 overflow-x-auto rounded-xl bg-slate-900 p-4 font-mono text-sm leading-relaxed text-slate-100" {...props}>
      {children}
    </pre>
  ),
  a: ({ ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a className="font-medium text-blue-700 underline hover:text-blue-900" {...props} />
  ),
  blockquote: ({ ...props }: React.HTMLAttributes<HTMLQuoteElement>) => (
    <blockquote className="mb-3 border-l-4 border-slate-300 pl-4 text-slate-700 italic" {...props} />
  ),
}

export function AnswerPanel({
  answer,
  response,
  isLoading,
  renderMarkdown,
}: {
  answer: string
  response: QueryResponseModel | null
  isLoading: boolean
  /** When false (streaming tokens), show plain text; when true, render markdown. */
  renderMarkdown: boolean
}) {
  const truthfulness = response?.truthfulness
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) {
      return
    }
    const t = window.setTimeout(() => setCopied(false), 2000)
    return () => window.clearTimeout(t)
  }, [copied])

  const handleCopy = useCallback(async () => {
    const text = answer.trim()
    if (!text) {
      return
    }
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch {
      /* clipboard unavailable */
    }
  }, [answer])

  const emptyPlaceholder = isLoading ? 'Waiting for tokens...' : 'Ask a question to see a grounded answer.'
  const showCopy = answer.trim().length > 0

  return (
    <section className="app-card p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold text-slate-950">Answer</h2>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            aria-label={copied ? 'Copied answer to clipboard' : 'Copy answer to clipboard'}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-800 hover:bg-slate-50 disabled:pointer-events-none disabled:opacity-40"
            disabled={!showCopy}
            onClick={() => void handleCopy()}
          >
            {copied ? (
              <>
                <Check className="h-4 w-4 shrink-0 text-emerald-600" aria-hidden="true" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-4 w-4 shrink-0" aria-hidden="true" />
                Copy
              </>
            )}
          </button>
          {truthfulness ? (
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-sm font-medium text-emerald-700">
              Truthfulness {truthfulness.score.toFixed(2)}
            </span>
          ) : null}
        </div>
      </div>
      <div
        className="min-h-28 rounded-xl bg-slate-50 p-4 text-left text-slate-800"
        aria-live={isLoading ? 'polite' : undefined}
      >
        {!answer && !isLoading ? (
          <p className="text-slate-600">{emptyPlaceholder}</p>
        ) : renderMarkdown && answer ? (
          <div className="answer-markdown">
            <ReactMarkdown components={markdownComponents}>{answer}</ReactMarkdown>
          </div>
        ) : (
          <div className="whitespace-pre-wrap leading-relaxed">{answer || emptyPlaceholder}</div>
        )}
      </div>
      {response ? (
        <div className="mt-3 flex flex-wrap gap-3 text-sm text-slate-600">
          <span>{response.provider} / {response.model}</span>
          <span>{Math.round(response.processing_time_ms)} ms</span>
          {response.cached ? <span>Cached</span> : null}
        </div>
      ) : null}
    </section>
  )
}
