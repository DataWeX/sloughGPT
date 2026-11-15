'use client'

/** The endpoint cards: params, examples, try-it runner and response output. */
import { CATEGORIES, ENDPOINTS, METHOD_COLORS } from './docs-endpoints'
import { PUBLIC_API_URL } from '@/lib/config'
import { Badge, Button, Card, CardContent, CardDescription, CardHeader } from '@sloughgpt/strui'
import type { DocsWorkbench } from './useDocsWorkbench'

export function DocsEndpointList({ wb }: { wb: DocsWorkbench }) {
  const {
    activeCategory,
    addToast,
    endpointKey,
    handleTryIt,
    loadingMap,
    responseRefs,
    responses,
    t,
  } = wb

  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text).catch(() => {})
  }
  return (
    <div className="flex-1 space-y-4">
      <h2 className="text-base font-medium">
        {t(CATEGORIES.find((c) => c.id === activeCategory)!.labelKey)}
      </h2>

      {ENDPOINTS[activeCategory].map((ep) => {
        const key = endpointKey(ep)
        const result = responses[key]
        const isLoading = loadingMap[key]

        return (
          <Card key={key}>
            <CardHeader>
              <div className="flex items-center gap-3">
                <Badge
                  variant="outline"
                  className={`font-mono text-xs border ${METHOD_COLORS[ep.method]}`}
                >
                  {ep.method}
                </Badge>
                <code className="text-sm font-mono">{ep.path}</code>
              </div>
              <CardDescription>{ep.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {ep.params && ep.params.length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-muted-foreground mb-1">
                    {t('consciousness_docs.params')}
                  </h4>
                  <div className="rounded-md border border-border/40 overflow-hidden">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-muted/30">
                          <th className="px-3 py-1.5 text-left font-medium">
                            {t('consciousness_docs.param_name')}
                          </th>
                          <th className="px-3 py-1.5 text-left font-medium">
                            {t('consciousness_docs.param_type')}
                          </th>
                          <th className="px-3 py-1.5 text-left font-medium">
                            {t('consciousness_docs.param_required')}
                          </th>
                          <th className="px-3 py-1.5 text-left font-medium">
                            {t('consciousness_docs.param_desc')}
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {ep.params.map((p) => (
                          <tr key={p.name} className="border-t border-border/30">
                            <td className="px-3 py-1.5 font-mono">{p.name}</td>
                            <td className="px-3 py-1.5 text-muted-foreground">{p.type}</td>
                            <td className="px-3 py-1.5">
                              {p.required ? (
                                <span className="text-warning">{t('consciousness_docs.yes')}</span>
                              ) : (
                                <span className="text-muted-foreground">
                                  {t('consciousness_docs.no')}
                                </span>
                              )}
                            </td>
                            <td className="px-3 py-1.5 text-muted-foreground">{p.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {ep.requestBody && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="text-xs font-medium text-muted-foreground">
                      {t('consciousness_docs.request_body')}
                    </h4>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-[10px]"
                      onClick={() => {
                        copyToClipboard(ep.requestBody!)
                        addToast(t('consciousness_docs.copied'), 'success')
                      }}
                    >
                      {t('consciousness_docs.copy')}
                    </Button>
                  </div>
                  <pre className="rounded-md border border-border/40 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-40 whitespace-pre-wrap">
                    {ep.requestBody}
                  </pre>
                </div>
              )}

              <div>
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-xs font-medium text-muted-foreground">
                    {t('consciousness_docs.response_example')}
                  </h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-[10px]"
                    onClick={() => {
                      copyToClipboard(ep.responseExample)
                      addToast(t('consciousness_docs.copied'), 'success')
                    }}
                  >
                    {t('consciousness_docs.copy')}
                  </Button>
                </div>
                <pre className="rounded-md border border-border/40 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-48 whitespace-pre-wrap">
                  {ep.responseExample}
                </pre>
              </div>

              <div className="flex items-center gap-2">
                <Button size="sm" onClick={() => handleTryIt(ep)} disabled={isLoading}>
                  {isLoading ? t('consciousness_docs.trying') : t('consciousness_docs.try_it')}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const code = `fetch("${PUBLIC_API_URL}${ep.path}", {\n  method: "${ep.method}",\n  headers: { "Content-Type": "application/json" }${ep.requestBody && ep.method !== 'GET' ? `,\n  body: JSON.stringify(${ep.requestBody})` : ''}\n}).then(r => r.json()).then(console.log)`
                    copyToClipboard(code)
                    addToast(t('consciousness_docs.copied'), 'success')
                  }}
                >
                  {t('consciousness_docs.copy_fetch')}
                </Button>
              </div>

              {result && (
                <div
                  ref={(el) => {
                    responseRefs.current[key] = el
                  }}
                >
                  <div className="flex items-center gap-3 text-xs mb-1">
                    <span
                      className={`font-mono font-medium ${
                        result.status >= 200 && result.status < 300
                          ? 'text-success'
                          : result.status >= 400
                            ? 'text-destructive'
                            : 'text-muted-foreground'
                      }`}
                    >
                      {result.status || 'ERR'}
                    </span>
                    <span className="text-muted-foreground">{result.timeMs}ms</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-5 px-1.5 text-[10px] ml-auto"
                      onClick={() => {
                        copyToClipboard(result.body)
                        addToast(t('consciousness_docs.copied'), 'success')
                      }}
                    >
                      {t('consciousness_docs.copy')}
                    </Button>
                  </div>
                  <pre className="rounded-md border border-border/40 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-64 whitespace-pre-wrap">
                    {result.body}
                  </pre>
                </div>
              )}
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
