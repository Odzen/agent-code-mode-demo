import { useState } from 'react'
import type { FormEvent } from 'react'
import { ArrowRight, Braces, ChevronDown, ExternalLink, ListTree, Ticket, X } from 'lucide-react'
import { createParser } from 'eventsource-parser'
import Markdown from 'react-markdown'

import { Badge } from '@/components/ui/badge'
import { Bubble, BubbleContent } from '@/components/ui/bubble'
import { Input } from '@/components/ui/input'
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupTextarea } from '@/components/ui/input-group'
import { Message, MessageContent } from '@/components/ui/message'
import {
  MessageScroller, MessageScrollerButton, MessageScrollerContent,
  MessageScrollerItem, MessageScrollerProvider, MessageScrollerViewport,
} from '@/components/ui/message-scroller'
import './App.css'

const example = 'Which customers have at least three overdue support tickets? Give me the counts.'
const approaches = [
  { id: 'tool-calling', name: 'Tool Calling', description: 'The model receives each page and decides what to call next.', Icon: ListTree },
  { id: 'code-mode', name: 'Code Mode', description: 'Python reads the pages inside Monty and returns the counts.', Icon: Braces },
] as const

type Approach = typeof approaches[number]['id']
type Model = 'deepseek-flash' | 'deepseek-v4-pro'
type Result = { answer: string; model_requests: number; pages_read: number; tool_output_bytes: number; input_tokens: number | null }
type Step = { name: string; detail?: string; code?: string }
type Panel = { text: string; trace: Step[]; result?: Result; error?: string }
type Panels = Record<Approach, Panel>

function emptyPanels(): Panels {
  return {
    'tool-calling': { text: '', trace: [] },
    'code-mode': { text: '', trace: [] },
  }
}

function Chat({ approach, panel, question, running }: {
  approach: typeof approaches[number]; panel: Panel; question: string; running: boolean
}) {
  const { Icon } = approach
  const result = panel.result
  return (
    <article className="chat" data-approach={approach.id} aria-label={approach.name}>
      <header className="chat-header">
        <div className="chat-icon"><Icon size={21} aria-hidden="true" /></div>
        <div><h2>{approach.name}</h2><p>{approach.description}</p></div>
      </header>
      <dl className="metrics">
        <div><dt>Model requests</dt><dd>{result?.model_requests ?? '–'}</dd></div>
        <div><dt>Ticket pages</dt><dd>{result?.pages_read ?? '–'}</dd></div>
        <div title="UTF-8 bytes from tool returns added to model history">
          <dt>Tool output bytes</dt><dd>{result?.tool_output_bytes.toLocaleString('en-US') ?? '–'}</dd>
        </div>
        {result?.input_tokens != null && <div><dt>Input tokens</dt><dd>{result.input_tokens.toLocaleString('en-US')}</dd></div>}
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
                          <BubbleContent className="answer"><Markdown>{panel.text}</Markdown></BubbleContent>
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
  const [model, setModel] = useState<Model>('deepseek-flash')
  const [apiKey, setApiKey] = useState('')
  const [prompt, setPrompt] = useState(example)
  const [question, setQuestion] = useState('')
  const [panels, setPanels] = useState<Panels>(emptyPanels)
  const [controller, setController] = useState<AbortController>()

  function updatePanel(approach: Approach, update: (panel: Panel) => Panel) {
    setPanels((previous) => ({ ...previous, [approach]: update(previous[approach]) }))
  }

  async function streamRun(approach: Approach, submitted: string, signal: AbortSignal) {
    try {
      const response = await fetch('/api/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey.trim()}`,
        },
        body: JSON.stringify({ approach, prompt: submitted, model }),
        signal,
      })
      if (response.status === 401) throw new Error('Enter a DeepSeek API key.')
      if (!response.ok || !response.body) throw new Error('The server could not start this run.')

      let finished = false
      const parser = createParser({
        onEvent({ event, data }) {
          if (event === 'tool') {
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
    if (!submitted || !apiKey.trim() || controller) return
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
        <div className="wordmark"><Braces size={19} aria-hidden="true" /> Agent Run Lab</div>
        <a href="https://github.com/Odzen/agent-code-mode-demo"><ExternalLink size={17} aria-hidden="true" /> Source</a>
      </nav>
      <section className="intro">
        <div>
          <h1>Same task. Two ways to run it.</h1>
          <p>Give both agents the same question. Follow the tool calls and see what reaches the model.</p>
        </div>
        <Badge variant="outline" className="dataset"><Ticket size={15} aria-hidden="true" /> 18 synthetic tickets · 3 pages</Badge>
      </section>
      <form onSubmit={compare} className="composer">
        <div className="provider-settings">
          <label>DeepSeek API key
            <Input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)}
              autoComplete="off" placeholder="Enter your DeepSeek API key" required disabled={!!controller} />
          </label>
          <label>Model
            <select value={model} onChange={(event) => setModel(event.target.value as Model)} disabled={!!controller}>
              <option value="deepseek-flash">DeepSeek Flash</option>
              <option value="deepseek-v4-pro">DeepSeek V4 Pro</option>
            </select>
          </label>
        </div>
        <label htmlFor="question" className="prompt-label">Ask both agents</label>
        <InputGroup className="composer-body">
          <InputGroupTextarea id="question" value={prompt} onChange={(event) => setPrompt(event.target.value)}
            readOnly={!!controller} maxLength={2000} rows={2} />
          <InputGroupAddon align="inline-end">
            {controller ? (
              <InputGroupButton type="button" className="submit" variant="outline" onClick={() => controller.abort()}>
                Stop <X size={15} aria-hidden="true" />
              </InputGroupButton>
            ) : (
              <InputGroupButton type="submit" className="submit" variant="default" disabled={!apiKey.trim() || !prompt.trim()}>
                Run both <ArrowRight size={15} aria-hidden="true" />
              </InputGroupButton>
            )}
          </InputGroupAddon>
        </InputGroup>
        <p className="composer-note">Both agents use your key for the selected model. The key stays in this tab’s memory between runs.</p>
      </form>
      <section className="comparison" aria-label="Agent comparison">
        {approaches.map((approach) => (
          <Chat key={approach.id} approach={approach} panel={panels[approach.id]} question={question} running={!!controller} />
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
