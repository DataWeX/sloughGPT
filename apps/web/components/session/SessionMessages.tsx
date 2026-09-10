import { Card, CardContent, CardHeader, CardTitle, cn } from '@sloughgpt/strui'

interface Message {
  role: string
  content: string
}

interface SessionMessagesProps {
  messages: Message[]
  title?: string
}

export function SessionMessages({ messages, title }: SessionMessagesProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">{title ?? `Recent Messages (${messages.length})`}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1 max-h-[240px] overflow-y-auto">
          {messages.map((msg, i) => (
            <div key={i} className={cn('rounded p-1.5 text-[10px]', msg.role === 'assistant' ? 'bg-primary/5' : 'bg-muted/30')}>
              <div className="flex items-center gap-1 mb-0.5">
                <span className={cn('text-[10px] font-medium', msg.role === 'assistant' ? 'text-primary' : 'text-muted-foreground/60')}>
                  {msg.role}
                </span>
              </div>
              <p className="whitespace-pre-wrap">{msg.content.slice(0, 500)}{msg.content.length > 500 ? '...' : ''}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
