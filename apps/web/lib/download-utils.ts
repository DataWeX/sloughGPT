'use client'

export function downloadBlob(data: Blob | BlobPart, filename: string, mimeType = 'application/octet-stream'): void {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function downloadJson(data: unknown, filename: string): void {
  const text = JSON.stringify(data, null, 2)
  downloadBlob(text, filename, 'application/json')
}

export function downloadMarkdown(content: string, filename: string): void {
  downloadBlob(content, filename, 'text/markdown')
}

export function importFile(accept: string): Promise<File | null> {
  return new Promise(resolve => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = accept
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0] ?? null
      resolve(file)
    }
    input.click()
  })
}
