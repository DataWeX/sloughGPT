'use client'

/**
 * Page chrome for Knowledge: the header actions (import/export/add) and
 * the search/sort/bulk toolbar. Pure render off the workbench.
 */
import {
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  IconChevronDown,
  IconDownload,
  IconMapPin,
  IconPlus,
  IconSearch,
  IconTrash,
  Input,
  Spinner,
} from '@sloughgpt/strui'
import type { KnowledgeWorkbench } from './useKnowledgeWorkbench'

export function KnowledgeHeaderActions({ wb }: { wb: KnowledgeWorkbench }) {
  const {
    fetchData,
    fileInputRef,
    items,
    handleExport,
    handleExportCSV,
    handleImportFile,
    importProgress,
    importing,
    loading,
    setShowAdd,
  } = wb

  return (
    <div className="flex items-center gap-2">
      <Button
        size="sm"
        variant="outline"
        className="h-7 text-xs"
        onClick={fetchData}
        disabled={loading}
      >
        <Spinner className="h-3 w-3 mr-1" />
        Refresh
      </Button>
      {items.length > 0 && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="sm" variant="outline" className="h-7 text-xs">
              <IconDownload className="h-3 w-3 mr-1" />
              Export
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-[140px]">
            <DropdownMenuItem onClick={handleExport}>Export as JSON</DropdownMenuItem>
            <DropdownMenuItem onClick={handleExportCSV}>Export as CSV</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
      <Button
        size="sm"
        variant="outline"
        className="h-7 text-xs"
        onClick={() => fileInputRef.current?.click()}
        disabled={importing}
      >
        {importing ? 'Importing...' : 'Import'}
      </Button>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,.txt,.csv"
        className="hidden"
        onChange={handleImportFile}
        aria-label="Import knowledge file"
      />
      {importProgress && importProgress.total > 0 && (
        <div className="flex items-center gap-2">
          <div className="w-24 h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${(importProgress.current / importProgress.total) * 100}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground font-mono">
            {importProgress.current}/{importProgress.total}
          </span>
        </div>
      )}
      <Button size="sm" className="h-7 text-xs" onClick={() => setShowAdd(true)}>
        <IconPlus className="h-3 w-3 mr-1" />
        Add
      </Button>
    </div>
  )
}

export function KnowledgeToolbar({ wb }: { wb: KnowledgeWorkbench }) {
  const {
    handleExportSelected,
    search,
    selectedIds,
    setPendingBatchDelete,
    setSearch,
    setShowBulkTopic,
    setSortBy,
    sortBy,
  } = wb

  return (
    <div className="flex items-center gap-2">
      <div className="relative flex-1">
        <IconSearch className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground/60" />
        <Input
          placeholder="Search knowledge..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-9 text-sm pl-9"
        />
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button size="sm" variant="outline" className="h-9 text-xs gap-1">
            {sortBy === 'date' ? 'Newest' : sortBy === 'importance' ? 'Importance' : 'Topic'}
            <IconChevronDown className="h-3 w-3" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={() => setSortBy('date')}>Newest</DropdownMenuItem>
          <DropdownMenuItem onClick={() => setSortBy('importance')}>Importance</DropdownMenuItem>
          <DropdownMenuItem onClick={() => setSortBy('topic')}>Topic</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {selectedIds.size > 0 && (
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            className="h-8 text-xs"
            onClick={handleExportSelected}
          >
            Export ({selectedIds.size})
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-8 text-xs"
            onClick={() => setShowBulkTopic(true)}
          >
            <IconMapPin className="h-3.5 w-3.5 mr-1" />
            Move to topic ({selectedIds.size})
          </Button>
          <Button
            size="sm"
            variant="destructive"
            className="h-8 text-xs"
            onClick={() => setPendingBatchDelete(true)}
          >
            <IconTrash className="h-3 w-3 mr-1" />
            Delete ({selectedIds.size})
          </Button>
        </div>
      )}
    </div>
  )
}
