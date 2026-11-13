import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { downloadBlob, downloadJson, downloadMarkdown, importFile } from './download-utils'

describe('download-utils', () => {
  let createObjectUrl: ReturnType<typeof vi.fn>
  let revokeObjectUrl: ReturnType<typeof vi.fn>
  let anchorClick: ReturnType<typeof vi.fn>
  let lastAnchor: HTMLAnchorElement | null
  let lastInput: HTMLInputElement | null

  beforeEach(() => {
    createObjectUrl = vi.fn(() => 'blob:mock-url')
    revokeObjectUrl = vi.fn()
    vi.stubGlobal('URL', { ...URL, createObjectURL: createObjectUrl, revokeObjectURL: revokeObjectUrl })
    anchorClick = vi.fn()
    HTMLAnchorElement.prototype.click = anchorClick as unknown as typeof HTMLAnchorElement.prototype.click

    lastAnchor = null
    lastInput = null
    const origCreate = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tag: string): HTMLElement => {
      const el = origCreate(tag)
      if (tag === 'a') lastAnchor = el as HTMLAnchorElement
      if (tag === 'input') lastInput = el as HTMLInputElement
      return el as HTMLElement
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  function captureAnchor() {
    return {
      href: lastAnchor?.getAttribute('href'),
      download: lastAnchor?.getAttribute('download'),
    }
  }

  it('downloads a Blob with the given filename', () => {
    const blob = new Blob(['hello'], { type: 'text/plain' })
    downloadBlob(blob, 'note.txt')
    const { href, download } = captureAnchor()
    expect(createObjectUrl).toHaveBeenCalledWith(blob)
    expect(href).toBe('blob:mock-url')
    expect(download).toBe('note.txt')
    expect(anchorClick).toHaveBeenCalledTimes(1)
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:mock-url')
  })

  it('wraps string content in a Blob with the provided mime type', () => {
    downloadBlob('content', 'data.txt', 'text/csv')
    const [blob] = createObjectUrl.mock.calls[0] as [Blob]
    expect(blob.type).toBe('text/csv')
    expect(captureAnchor().download).toBe('data.txt')
  })

  it('downloadJson serializes data as pretty-printed JSON', async () => {
    downloadJson({ a: 1 }, 'export.json')
    const [blob] = createObjectUrl.mock.calls[0] as [Blob]
    expect(blob.type).toBe('application/json')
    await expect(blob.text()).resolves.toBe('{\n  "a": 1\n}')
  })

  it('downloadMarkdown uses the markdown mime type', () => {
    downloadMarkdown('# Title', 'readme.md')
    const [blob] = createObjectUrl.mock.calls[0] as [Blob]
    expect(blob.type).toBe('text/markdown')
    expect(captureAnchor().download).toBe('readme.md')
  })

  it('importFile resolves with the selected file on change', async () => {
    const file = new File(['x'], 'data.json', { type: 'application/json' })
    const promise = importFile('.json')
    Object.defineProperty(lastInput!, 'files', { value: [file] })
    lastInput!.dispatchEvent(new Event('change'))
    await expect(promise).resolves.toBe(file)
    expect(lastInput!.accept).toBe('.json')
  })

  it('importFile resolves with null when no file is chosen', async () => {
    const promise = importFile('*')
    Object.defineProperty(lastInput!, 'files', { value: [] })
    lastInput!.dispatchEvent(new Event('change'))
    await expect(promise).resolves.toBeNull()
  })
})
