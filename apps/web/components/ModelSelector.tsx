'use client'

import { IconRefresh } from '@/components/ui'
import { useModels, useLocalModels, useHuggingFaceModels } from '@/contexts/ModelContext'
import { cn } from '@/lib/cn'
import { Select } from '@/components/ui/select'
import { Button } from '@/components/ui/button'

interface ModelSelectorProps {
  value?: string
  onValueChange?: (modelId: string) => void
  filter?: 'all' | 'local' | 'huggingface'
  showLoadButton?: boolean
  className?: string
  placeholder?: string
}

export function ModelSelector({
  value,
  onValueChange,
  filter = 'all',
  showLoadButton = true,
  className,
  placeholder = 'Select a model',
}: ModelSelectorProps) {
  const { models, loading, loadingModelId, loadModel, currentModel, isModelLoaded } = useModels()
  const localModels = useLocalModels()
  const hfModels = useHuggingFaceModels()

  const filteredModels = filter === 'all' ? models
    : filter === 'local' ? localModels
    : hfModels

  const displayModels = filteredModels.length > 0 ? filteredModels : models

  const options = displayModels.map(m => {
    // Clean up the display name - remove "local/" prefix, truncate long names
    let label = m.name || m.id
    if (label.startsWith('local/')) {
      label = label.slice(6)
    }
    // Truncate if too long
    if (label.length > 40) {
      label = label.slice(0, 37) + '...'
    }
    return {
      value: m.id,
      label,
    }
  })

  const handleLoadModel = async (modelId: string) => {
    const result = await loadModel(modelId, { mode: 'local' })
    if (result.success && onValueChange) {
      onValueChange(modelId)
    }
  }

  const cleanModelName = (id: string) => {
    let name = id
    if (name.startsWith('local/')) name = name.slice(6)
    if (name.length > 40) name = name.slice(0, 37) + '...'
    return name
  }

  const currentLabel = value ? cleanModelName(value) : ''

  return (
    <div className={cn("flex gap-2", className)}>
      <Select
        value={value || currentModel || ''}
        onValueChange={onValueChange || (() => {})}
        options={options}
        placeholder={placeholder}
        className="w-[200px]"
      />

      {showLoadButton && value && (
        <Button
          size="sm"
          onClick={() => handleLoadModel(value)}
          disabled={loading || value === currentModel}
          className="gap-1.5"
        >
          {loading && loadingModelId === value ? (
            <>
              <IconRefresh className="h-3 w-3 animate-spin" />
              Loading...
            </>
          ) : value === currentModel && isModelLoaded ? (
            'Loaded ✓'
          ) : (
            <>Load</>
          )}
        </Button>
      )}
    </div>
  )
}

interface ModelCardProps {
  model: {
    id: string
    name: string
    type: string
    sizeMb?: number
    params?: string
    description?: string
    tags?: string[]
  }
  isActive?: boolean
  isLoading?: boolean
  onLoad?: () => void
  onSelect?: () => void
  className?: string
}

export function ModelCard({
  model,
  isActive = false,
  isLoading = false,
  onLoad,
  onSelect,
  className,
}: ModelCardProps) {
  return (
    <article
      className={cn(
        "p-4 rounded-lg border transition-all cursor-pointer",
        isActive
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50 hover:bg-muted/50",
        className
      )}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onSelect?.()
        }
      }}
      aria-label={`${model.name}${model.type === 'local' ? ', local model' : ', Hugging Face model'}${model.sizeMb ? `, ${model.sizeMb < 1 ? `${(model.sizeMb * 1024).toFixed(0)} KB` : `${model.sizeMb.toFixed(1)} MB`}` : ''}`}
    >
      <div className="flex items-start justify-between mb-2">
        <h3 className="font-medium">{model.name}</h3>
        <span className={cn(
          "text-xs px-2 py-0.5 rounded-full",
          model.type === 'local'
            ? "bg-success/10 text-success"
            : "bg-primary/10 text-primary"
        )}>
          {model.type}
        </span>
      </div>

      {model.description && (
        <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
          {model.description}
        </p>
      )}

      <div className="flex items-center gap-3 text-xs text-muted-foreground mb-3">
        {model.sizeMb && (
          <span>{model.sizeMb < 1 ? `${(model.sizeMb * 1024).toFixed(0)} KB` : `${model.sizeMb.toFixed(1)} MB`}</span>
        )}
        {model.params && <span>{model.params}</span>}
      </div>

      {model.tags && model.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3" role="list" aria-label="Tags">
          {model.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground"
              role="listitem"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      <div className="flex justify-end">
        {isActive ? (
          <span className="text-sm text-success font-medium" aria-live="polite">
            Active
          </span>
        ) : onLoad ? (
          <Button
            size="sm"
            variant={isLoading ? "secondary" : "default"}
            onClick={(e) => {
              e.stopPropagation()
              onLoad()
            }}
            disabled={isLoading}
            aria-busy={isLoading}
          >
            {isLoading ? 'Loading...' : 'Load'}
          </Button>
        ) : null}
      </div>
    </article>
  )
}