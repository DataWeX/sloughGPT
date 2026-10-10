'use client'

export const dynamic = 'force-dynamic'

import { PageContainer } from '@/components/PageContainer'
import { Button, Input, KpiGrid, StatCard, SectionHeader } from '@sloughgpt/strui'
import { Download, Search, X, Tag, Trash2, Star } from 'lucide-react'
import { useTrainingRuns } from '@/hooks/useTrainingRuns'
import { RunList } from '@/components/training/RunList'
import { RunDetailsPanel } from '@/components/training/RunDetailsPanel'

export default function TrainingRunsPage() {
  const {
    runs,
    filteredRuns,
    loading,
    models,
    methods,
    summary,
    searchQuery,
    setSearchQuery,
    filterModel,
    setFilterModel,
    filterMethod,
    setFilterMethod,
    format,
    setFormat,
    handleExport,
    selectedIds,
    toggleSelect,
    toggleSelectAll,
    clearSelection,
    bulkTag,
    setBulkTag,
    handleBulkTag,
    handleBulkBookmark,
    handleBulkDelete,
    deleting,
    selectedRun,
    selectRun,
    closeSelection,
    handleBookmark,
    handleDuplicate,
    handleDelete,
    handleExportSingle,
    newTag,
    setNewTag,
    handleAddTag,
    handleRemoveTag,
    editingNotes,
    startEditNotes,
    setEditingNotes,
    notesValue,
    setNotesValue,
    handleSaveNotes,
    compareRunId,
    changeCompareTarget,
    compareResult,
    handleCompare,
  } = useTrainingRuns()

  return (
    <PageContainer
      title="Training Runs"
      headerRight={
        <div className="flex items-center gap-2">
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value as 'json' | 'csv')}
            className="text-sm border rounded px-2 py-1"
            aria-label="Export format"
          >
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
          </select>
          <Button size="sm" variant="outline" onClick={handleExport}>
            <Download className="h-4 w-4 mr-1" /> Export
          </Button>
        </div>
      }
      toolbar={
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search runs, tags, notes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8"
              aria-label="Search runs"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-2.5"
                aria-label="Clear search"
              >
                <X className="h-4 w-4 text-muted-foreground" />
              </button>
            )}
          </div>
          <select
            value={filterModel}
            onChange={(e) => setFilterModel(e.target.value)}
            className="text-sm border rounded px-2 py-1.5"
            aria-label="Filter by model"
          >
            <option value="">All Models</option>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <select
            value={filterMethod}
            onChange={(e) => setFilterMethod(e.target.value)}
            className="text-sm border rounded px-2 py-1.5"
            aria-label="Filter by method"
          >
            <option value="">All Methods</option>
            {methods.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
      }
    >
      <SectionHeader
        title="Run history"
        description={`${filteredRuns.length} of ${runs.length} shown`}
      />

      <KpiGrid columns={4}>
        <StatCard label="Total Runs" value={summary.total} numeric />
        <StatCard label="Converged" value={summary.converged} numeric />
        <StatCard label="Avg Quality" value={`${Math.round(summary.avgQuality * 100)}%`} numeric />
        <StatCard label="Avg Loss" value={summary.avgLoss.toFixed(4)} numeric />
      </KpiGrid>

      {selectedIds.size > 0 && (
        <div className="flex items-center gap-2 mb-4 p-3 bg-muted rounded-lg">
          <span className="text-sm font-medium">{selectedIds.size} selected</span>
          <div className="flex items-center gap-1 ml-2">
            <Input
              placeholder="Tag name..."
              value={bulkTag}
              onChange={(e) => setBulkTag(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleBulkTag()}
              className="w-32 text-xs h-7"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs"
              onClick={handleBulkTag}
              disabled={!bulkTag.trim()}
            >
              <Tag className="h-3 w-3 mr-1" /> Tag
            </Button>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs"
            onClick={() => handleBulkBookmark(true)}
          >
            <Star className="h-3 w-3 mr-1" /> Bookmark
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs"
            onClick={() => handleBulkBookmark(false)}
          >
            Unbookmark
          </Button>
          <Button
            size="sm"
            variant="destructive"
            className="h-7 text-xs"
            onClick={handleBulkDelete}
          >
            <Trash2 className="h-3 w-3 mr-1" /> Delete
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs ml-auto"
            onClick={clearSelection}
          >
            Clear Selection
          </Button>
        </div>
      )}

      <div className="flex gap-4">
        <RunList
          loading={loading}
          runs={filteredRuns}
          selectedRunId={selectedRun?.run_id}
          selectedIds={selectedIds}
          deleting={deleting}
          onSelect={selectRun}
          onToggleSelect={toggleSelect}
          onToggleSelectAll={toggleSelectAll}
          onBookmark={handleBookmark}
          onDuplicate={handleDuplicate}
          onDelete={handleDelete}
        />

        {selectedRun && (
          <RunDetailsPanel
            run={selectedRun}
            runs={runs}
            onBookmark={handleBookmark}
            onExportSingle={handleExportSingle}
            onClose={closeSelection}
            newTag={newTag}
            onNewTagChange={setNewTag}
            onAddTag={handleAddTag}
            onRemoveTag={handleRemoveTag}
            editingNotes={editingNotes}
            onEditNotes={startEditNotes}
            notesValue={notesValue}
            onNotesChange={setNotesValue}
            onSaveNotes={handleSaveNotes}
            onCancelEdit={() => setEditingNotes(false)}
            compareRunId={compareRunId}
            onCompareIdChange={changeCompareTarget}
            compareResult={compareResult}
            onCompare={handleCompare}
          />
        )}
      </div>
    </PageContainer>
  )
}
