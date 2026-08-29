'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Textarea, Input, Button, Chip } from '@sloughgpt/strui'
import { IconChevronRight, IconChevronLeft } from '@sloughgpt/strui'
import { tokenTreeController, type EncodeResult } from '@/lib/token-tree-controller'
import { useToastStore } from '@/lib/toast-store'

const parseIds = (input: string): number[] =>
  input
    .split(/[,\s]+/)
    .filter(part => part.length > 0)
    .map(part => parseInt(part, 10))
    .filter(id => Number.isFinite(id))

const displayToken = (token: string) => token.replace('</w>', '').trim() || token

export function TokenTreeCodecCard() {
  const [text, setText] = useState('the quick brown fox jumps over the lazy dog')
  const [encodeResult, setEncodeResult] = useState<EncodeResult | null>(null)
  const [encoding, setEncoding] = useState(false)
  const [idsInput, setIdsInput] = useState('')
  const [decodeText, setDecodeText] = useState('')
  const [decoding, setDecoding] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const handleEncode = async () => {
    const term = text.trim()
    if (!term) return
    setEncoding(true)
    try {
      const result = await tokenTreeController.encode(term)
      setEncodeResult(result)
      setIdsInput(result.ids.join(', '))
      setDecodeText('')
    } catch {
      addToast('Could not encode text', 'error')
    } finally {
      setEncoding(false)
    }
  }

  const handleDecode = async () => {
    const ids = parseIds(idsInput)
    if (ids.length === 0) {
      addToast('Enter at least one token id', 'error')
      return
    }
    setDecoding(true)
    try {
      const result = await tokenTreeController.decode(ids)
      setDecodeText(result.text)
    } catch {
      addToast('Could not decode ids', 'error')
    } finally {
      setDecoding(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Token Tree Codec</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Encode text
          </label>
          <Textarea
            value={text}
            onChange={e => setText(e.target.value)}
            rows={2}
            placeholder="Text to tree-walk encode"
            aria-label="Text to encode"
          />
          <div>
            <Button size="sm" onClick={handleEncode} disabled={encoding || !text.trim()}>
              {encoding ? 'Encoding...' : (
                <>
                  <IconChevronRight className="h-4 w-4 mr-1" />
                  Encode
                </>
              )}
            </Button>
          </div>
        </div>

        {encodeResult && (
          <div className="rounded-md bg-muted/50 px-3 py-2 space-y-2">
            <div className="text-xs text-muted-foreground">{encodeResult.ids.length} tokens</div>
            <div className="flex flex-wrap gap-1">
              {encodeResult.tokens.map((token, i) => (
                <Chip key={i} label={displayToken(token)} />
              ))}
            </div>
            <div className="text-xs font-mono break-all text-muted-foreground">
              [{encodeResult.ids.join(', ')}]
            </div>
          </div>
        )}

        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Decode ids
          </label>
          <div className="flex items-center gap-2">
            <Input
              value={idsInput}
              onChange={e => setIdsInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') handleDecode()
              }}
              placeholder="Comma-separated token ids, e.g. 3, 12"
              className="max-w-xs"
              aria-label="Token ids to decode"
            />
            <Button size="sm" onClick={handleDecode} disabled={decoding}>
              {decoding ? 'Decoding...' : (
                <>
                  <IconChevronLeft className="h-4 w-4 mr-1" />
                  Decode
                </>
              )}
            </Button>
          </div>
        </div>

        {decodeText && (
          <div className="rounded-md bg-success/10 border border-success/20 px-3 py-2 text-sm">
            <span className="text-muted-foreground">Decoded: </span>
            <span className="font-mono">"{decodeText}"</span>
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          Encode walks the trained merge tree top-down; decode maps token ids back to text. Ids from an encode are
          pre-filled below so you can verify the round trip in one click.
        </p>
      </CardContent>
    </Card>
  )
}
