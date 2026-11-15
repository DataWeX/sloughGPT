import { Card, CardContent, Button, Input } from '@sloughgpt/strui'

export interface NotificationFiltersProps {
  filter: string
  onFilterChange: (value: string) => void
  onRefresh: () => void
}

export function NotificationFilters({
  filter,
  onFilterChange,
  onRefresh,
}: NotificationFiltersProps) {
  return (
    <Card className="mb-4">
      <CardContent className="py-2">
        <div className="flex gap-2">
          <Input
            value={filter}
            onChange={(e) => onFilterChange(e.target.value)}
            placeholder="Filter notifications..."
            className="flex-1 h-6 text-xs"
          />
          <Button size="sm" variant="outline" className="h-6 text-xs" onClick={onRefresh}>
            Refresh
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
