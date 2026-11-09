import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { ArrowUp, Braces, ChevronDown, ExternalLink, ListTree, Square, Ticket } from 'lucide-react'
import { createParser } from 'eventsource-parser'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { Badge } from '@/components/ui/badge'
import { Bubble, BubbleContent } from '@/components/ui/bubble'
import { Button } from '@/components/ui/button'
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupText, InputGroupTextarea } from '@/components/ui/input-group'
import { Message, MessageContent } from '@/components/ui/message'
import {
  MessageScroller, MessageScrollerButton, MessageScrollerContent,
  MessageScrollerItem, MessageScrollerProvider, MessageScrollerViewport,
} from '@/components/ui/message-scroller'
import './App.css'

const examples = [
  { label: '3+ overdue', prompt: 'Which customers have at least three overdue support tickets? Give me the counts.' },
  { label: 'By customer', prompt: 'Read all pages and rank customers by their number of overdue tickets.' },
  { label: 'Password resets', prompt: 'Read all pages. Which customers have overdue password reset tickets? List their ticket IDs.' },
  { label: 'Totals', prompt: 'How many overdue tickets and distinct customers are there across all pages?' },
  { label: 'Enterprise SLAs', prompt: 'Read all ticket pages. For customers with at least three overdue tickets, look up their profiles and show the Enterprise accounts with their overdue counts and response SLAs.' },
  { label: 'Code Mode issues', prompt: 'Search public issues in pydantic/pydantic-ai once, using the combined query run_code Monty. From those results, show two relevant issues with their titles, open or closed status, and GitHub links.' },
]
const approaches = [
  { id: 'tool-calling', name: 'Tool Calling', description: 'The model receives each page and decides what to call next.', Icon: ListTree },
  { id: 'code-mode', name: 'Code Mode', description: 'Python combines tool results inside Monty and returns the result.', Icon: Braces },
] as const

type Approach = typeof approaches[number]['id']
type Result = { answer: string; model_requests: number; pages_read: number; tool_output_bytes: number; input_tokens: number | null; output_tokens?: number; elapsed_ms?: number }
type Step = { name: string; detail?: string; code?: string }
type Panel = { text: string; trace: Step[]; result?: Result; error?: string }
type Panels = Record<Approach, Panel>

function emptyPanels(): Panels {
  return {
    'tool-calling': { text: '', trace: [] },
    'code-mode': { text: '', trace: [] },
  }
}

function Chat({ approach, panel, question, running, tools }: {
  approach: typeof approaches[number]; panel: Panel; question: string; running: boolean; tools: string[]
}) {
  const { Icon } = approach
  const result = panel.result
  return (
    <article className="chat" data-approach={approach.id} aria-label={approach.name}>
      <header className="chat-header">
        <div className="chat-icon"><Icon size={21} aria-hidden="true" /></div>
        <div><h2>{approach.name}</h2><p>{approach.description}</p></div>
      </header>
      <div className="tool-catalog" role="group" aria-label="Available tools">
        <span>Tools</span>
        {approach.id === 'code-mode' && <><code>run_code</code><span>Inside sandbox</span></>}
        {tools.map((tool) => <code key={tool}>{tool}</code>)}
      </div>
      <dl className="metrics">
        <div title="Server time from the start of the run to the final response">
          <dt>Time</dt><dd>{result?.elapsed_ms != null ? `${(result.elapsed_ms / 1000).toFixed(1)} s` : '–'}</dd>
        </div>
        <div><dt>Model requests</dt><dd>{result?.model_requests ?? '–'}</dd></div>
        <div><dt>Ticket pages</dt><dd>{result?.pages_read ?? '–'}</dd></div>
        <div title="UTF-8 bytes from tool returns added to model history">
          <dt>Tool output bytes</dt><dd>{result?.tool_output_bytes.toLocaleString('en-US') ?? '–'}</dd>
        </div>
        {result?.input_tokens != null && <div><dt>Input tokens</dt><dd>{result.input_tokens.toLocaleString('en-US')}</dd></div>}
        {result?.output_tokens != null && <div><dt>Output tokens</dt><dd>{result.output_tokens.toLocaleString('en-US')}</dd></div>}
      </dl>
      {question ? (
        <MessageScrollerProvider autoScroll>
          <MessageScroller className="chat-scroll">
            <MessageScrollerViewport>
              <MessageScrollerContent className="gap-3 p-4">
                <MessageScrollerItem messageId="question" scrollAnchor>
                  <Message align="end">
                    <MessageContent>
                      <Bubble align="end" variant="muted">
                        <BubbleContent className="question">{question}</BubbleContent>
                      </Bubble>
                    </MessageContent>
                  </Message>
                </MessageScrollerItem>
                <MessageScrollerItem messageId="answer">
                  <Message>
                    <MessageContent>
                      {panel.trace.length > 0 && (
                        <details className="trace">
                          <summary>
                            <ListTree size={15} aria-hidden="true" />
                            <span>Execution trace</span>
                            <span className="trace-count">{panel.trace.length} steps</span>
                            <ChevronDown className="chevron" size={15} aria-hidden="true" />
                          </summary>
                          <ol>
                            {panel.trace.map((step, index) => (
                              <li key={index}>
                                <div><code>{step.name}</code>{step.detail && <span>{step.detail}</span>}</div>
                                {step.code && <pre><code>{step.code}</code></pre>}
                              </li>
                            ))}
                          </ol>
                        </details>
                      )}
                      {panel.text && (
                        <Bubble variant="muted">
                          <BubbleContent className="answer"><Markdown remarkPlugins={[remarkGfm]}>{panel.text}</Markdown></BubbleContent>
                        </Bubble>
                      )}
                      {panel.error && <p role="alert" className="run-error">{panel.error}</p>}
                      {running && !result && !panel.error && <p role="status" className="running">Running agent…</p>}
                    </MessageContent>
                  </Message>
                </MessageScrollerItem>
              </MessageScrollerContent>
            </MessageScrollerViewport>
            <MessageScrollerButton />
          </MessageScroller>
        </MessageScrollerProvider>
      ) : (
        <div className="empty">
          <Icon size={25} strokeWidth={1.5} aria-hidden="true" />
          <p>The response and tool calls will appear here.</p>
        </div>
      )}
    </article>
  )
}

function App() {
  const [config, setConfig] = useState<{ model: string; ready: boolean; tools: string[] }>()
  const [prompt, setPrompt] = useState(examples[0].prompt)
  const [question, setQuestion] = useState('')
  const [panels, setPanels] = useState<Panels>(emptyPanels)
  const [controller, setController] = useState<AbortController>()

  useEffect(() => {
    const aborter = new AbortController()
    fetch('/api/config', { signal: aborter.signal })
      .then(async (response) => {
        if (response.ok) {
          setConfig(await response.json())
        }
      })
      .catch(() => { /* Keep the demo disabled when configuration is unavailable. */ })
    return () => aborter.abort()
  }, [])

  function updatePanel(approach: Approach, update: (panel: Panel) => Panel) {
    setPanels((previous) => ({ ...previous, [approach]: update(previous[approach]) }))
  }

  async function streamRun(approach: Approach, submitted: string, signal: AbortSignal) {
    try {
      const response = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approach, prompt: submitted }),
        signal,
      })
      if (response.status === 503) throw new Error('The demo is waiting for its Hugging Face token.')
      if (!response.ok || !response.body) throw new Error('The server could not start this run.')

      let finished = false
      const parser = createParser({
        onEvent({ event, data }) {
          if (event === 'tool' || event === 'activity') {
            const tool: Step = JSON.parse(data)
            updatePanel(approach, (panel) => ({ ...panel, trace: [...panel.trace, tool] }))
          } else if (event === 'page') {
            const page: { cursor: number | null; items: number } = JSON.parse(data)
            const step = { name: 'GET /api/tickets', detail: `cursor ${page.cursor ?? 0}, ${page.items} tickets` }
            updatePanel(approach, (panel) => ({ ...panel, trace: [...panel.trace, step] }))
          } else if (event === 'text') {
            const text: string = JSON.parse(data)
            updatePanel(approach, (panel) => ({ ...panel, text: panel.text + text }))
          } else if (event === 'done') {
            const result: Result = JSON.parse(data)
            finished = true
            updatePanel(approach, (panel) => ({ ...panel, text: result.answer, result }))
          } else if (event === 'error') {
            finished = true
            const error: string = JSON.parse(data)
            updatePanel(approach, (panel) => ({ ...panel, error }))
          }
        },
      })
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          parser.feed(value)
        }
      } finally {
        reader.releaseLock()
      }
      if (!finished) throw new Error('The connection closed before this run finished.')
    } catch (error) {
      const message = signal.aborted
        ? 'Run stopped.'
        : error instanceof Error ? error.message : 'The run failed. Try again.'
      updatePanel(approach, (panel) => panel.result ? panel : { ...panel, error: message })
    }
  }

  async function compare(event: FormEvent) {
    event.preventDefault()
    const submitted = prompt.trim()
    if (!submitted || !config?.ready || controller) return
    const aborter = new AbortController()
    setController(aborter)
    setQuestion(submitted)
    setPanels(emptyPanels())
    try {
      await Promise.all(approaches.map(({ id }) => streamRun(id, submitted, aborter.signal)))
    } finally {
      setController(undefined)
    }
  }

  return (
    <main className="lab">
      <nav className="lab-nav" aria-label="Lab navigation">
        <h1 className="wordmark"><Braces size={19} aria-hidden="true" /> Agent Run Lab</h1>
        <a href="https://github.com/Odzen/agent-code-mode-demo"><ExternalLink size={17} aria-hidden="true" /> Source</a>
      </nav>
      <div className="example-bar">
        <div className="examples" role="group" aria-label="Example questions">
          <span>Try an example</span>
          {examples.map(({ label, prompt: examplePrompt }) => (
            <Button key={label} type="button" size="sm" variant="outline"
              aria-pressed={prompt === examplePrompt} disabled={!!controller} onClick={() => setPrompt(examplePrompt)}>
              {label}
            </Button>
          ))}
        </div>
        <Badge variant="outline" className="dataset"><Ticket size={15} aria-hidden="true" /> 18 synthetic tickets · 3 pages</Badge>
      </div>
      <form onSubmit={compare} className="composer">
        <label htmlFor="question" className="sr-only">Ask both agents</label>
        <InputGroup className="rounded-2xl bg-card">
          <InputGroupTextarea
            id="question"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }}
            readOnly={!!controller}
            maxLength={2000}
            rows={2}
            placeholder="Ask both agents about the ticket dataset…"
            className="p-3"
          />
          <InputGroupAddon align="block-end">
            <InputGroupText title={config?.model}>DeepSeek Flash</InputGroupText>
            {controller ? (
              <InputGroupButton type="button" size="icon-sm" className="ml-auto rounded-full"
                variant="outline" aria-label="Stop both agents" title="Stop both agents" onClick={() => controller.abort()}>
                <Square size={15} aria-hidden="true" />
              </InputGroupButton>
            ) : (
              <InputGroupButton type="submit" size="icon-sm" className="ml-auto rounded-full"
                variant="default" aria-label="Run both agents" title="Run both agents" disabled={!config?.ready || !prompt.trim()}>
                <ArrowUp size={16} aria-hidden="true" />
              </InputGroupButton>
            )}
          </InputGroupAddon>
        </InputGroup>
        {config?.ready === false && (
          <p role="status" className="composer-note">The demo is waiting for its Hugging Face token.</p>
        )}
      </form>
      <section className="comparison" aria-label="Agent comparison">
        {approaches.map((approach) => (
          <Chat key={approach.id} approach={approach} panel={panels[approach.id]} question={question} running={!!controller} tools={config?.tools ?? []} />
        ))}
      </section>
      <footer className="lab-footer">
        <span>Each comparison starts with a fresh conversation.</span>
        <span>FastAPI · Pydantic AI · Monty</span>
      </footer>
    </main>
  )
}

export default App
