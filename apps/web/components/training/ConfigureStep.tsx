'use client'

import { useMemo, useEffect, useState } from 'react'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  ToggleGroup,
  ToggleGroupItem,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  Input,
  Label,
  Checkbox,
  FoldSection,
} from '@sloughgpt/strui'
import { TrainingPresets } from '@/components/training/TrainingPresets'
import {
  trainingJobsController,
  type TrainingRecommendationResponse,
} from '@/lib/training-controller'
import {
  TRAINING_LIMITS,
  clampTrainingField,
  parseTrainingField,
  validateTrainingField,
} from '@/lib/sanitize'
import type { StepProps } from './DataStep'

export function ConfigureStep({ form, datasets, checkpoints, onNext, onBack }: StepProps) {
  const [recommendation, setRecommendation] = useState<TrainingRecommendationResponse | null>(null)
  const [loadingRec, setLoadingRec] = useState(false)

  // Validation derives from TRAINING_LIMITS (single source of truth).
  // NaN is explicitly invalid: every old-style `< min || > max` comparison
  // against NaN is false, so a cleared field used to pass validation.
  const hp = useMemo(
    () => ({
      epochs: validateTrainingField('epochs', form.trainingEpochs, 'Epochs'),
      batch: validateTrainingField('batch_size', form.trainingBatchSize, 'Batch size'),
      lr: validateTrainingField('learning_rate', form.trainingLR, 'Learning rate'),
    }),
    [form.trainingEpochs, form.trainingBatchSize, form.trainingLR],
  )
  const hpErrors = useMemo(
    () => [hp.epochs, hp.batch, hp.lr].filter((e): e is string => e !== null),
    [hp],
  )

  // The hyperparameter inputs hold a local draft string so typing never
  // fights the controlled value — `0.001` stays typeable — and the parsed,
  // clamped commit happens exactly once, on blur (reported bug: the old
  // `Number(x) || 1e-3` onChange snapped the field on every keystroke).
  const [epochsDraft, setEpochsDraft] = useState(String(form.trainingEpochs))
  const [batchDraft, setBatchDraft] = useState(String(form.trainingBatchSize))
  const [lrDraft, setLrDraft] = useState(String(form.trainingLR))

  useEffect(() => {
    setEpochsDraft(String(form.trainingEpochs))
  }, [form.trainingEpochs])
  useEffect(() => {
    setBatchDraft(String(form.trainingBatchSize))
  }, [form.trainingBatchSize])
  useEffect(() => {
    setLrDraft(String(form.trainingLR))
  }, [form.trainingLR])

  const commitField = (
    key: 'epochs' | 'batch_size' | 'learning_rate',
    raw: string,
    current: number,
    setDraft: (s: string) => void,
    set: (n: number) => void,
  ) => {
    const next = parseTrainingField(key, raw)
    if (Number.isFinite(next)) {
      setDraft(String(next))
      if (next !== current) set(next)
    } else {
      // Empty or unparseable — revert the draft to the last valid value.
      setDraft(String(current))
    }
  }

  const canAdvance = form.canStart && hpErrors.length === 0

  // Fetch recommendation when dataset changes
  useEffect(() => {
    if (datasets?.selectedDataset) {
      setLoadingRec(true)
      trainingJobsController
        .getTrainingRecommendation(datasets.selectedDataset, form.method)
        .then(setRecommendation)
        .catch(() => setRecommendation(null))
        .finally(() => setLoadingRec(false))
    }
  }, [datasets?.selectedDataset, form.method])

  const applyRecommendation = () => {
    if (!recommendation) return
    const rec = recommendation.recommendation
    // Clamp: backend recommendations must not smuggle out-of-range values
    // into the form (single source of truth = TRAINING_LIMITS).
    form.setTrainingLR(clampTrainingField('learning_rate', rec.learning_rate))
    form.setTrainingBatchSize(clampTrainingField('batch_size', rec.batch_size))
    form.setTrainingEpochs(clampTrainingField('epochs', rec.epochs))
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">2. Configure training</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Top-level mode: Text or Vision */}
        <div className="flex items-center gap-1" role="radiogroup" aria-label="Training mode">
          <ToggleGroup
            type="single"
            value={form.method === 'vlm' ? 'vlm' : 'text'}
            onValueChange={(v) => {
              if (v) form.setMethod(v === 'vlm' ? 'vlm' : 'distill')
            }}
          >
            <ToggleGroupItem
              value="text"
              className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
            >
              Text
            </ToggleGroupItem>
            <ToggleGroupItem
              value="vlm"
              className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
            >
              Vision
            </ToggleGroupItem>
          </ToggleGroup>
        </div>

        <div className="text-[10px] text-muted-foreground/60">
          {form.method === 'vlm'
            ? 'Teach the AI to understand images and text'
            : 'Train a model on text data'}
        </div>

        <TrainingPresets
          onApply={form.applyPreset}
          customPresets={form.customPresets}
          onSave={form.saveCustomPreset}
          onDelete={form.deleteCustomPreset}
          getCurrentState={() => ({
            name: '',
            description: '',
            method: form.method as 'distill' | 'finetune' | 'native' | 'vlm',
            epochs: form.trainingEpochs,
            lr: form.trainingLR,
            batchSize: form.trainingBatchSize,
            useLoRA: form.useLoRA,
            nativeEmbed: form.nativeEmbed,
            nativeLayers: form.nativeLayers,
            nativeHeads: form.nativeHeads,
            nativeBlockSize: form.nativeBlockSize,
          })}
        />

        {form.method !== 'vlm' ? (
          <>
            {/* Text training sub-method */}
            <div className="flex items-center gap-1" role="radiogroup" aria-label="Training method">
              <ToggleGroup
                type="single"
                value={form.method}
                onValueChange={(v) => {
                  if (v) form.setMethod(v as 'distill' | 'finetune' | 'native')
                }}
              >
                <ToggleGroupItem
                  value="distill"
                  className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
                >
                  Train from scratch
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="finetune"
                  className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
                >
                  Continue training
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="native"
                  className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
                >
                  Native SloNet
                </ToggleGroupItem>
              </ToggleGroup>
            </div>

            <div className="text-[10px] text-muted-foreground/60">
              {form.method === 'distill' &&
                'Train a small model from text data — no teacher needed'}
              {form.method === 'finetune' && 'Continue training an existing model on new data'}
              {form.method === 'native' &&
                'Train a pure transformer from scratch — SloNet architecture'}
            </div>

            {/* Data source */}
            <div className="flex items-center gap-1" role="radiogroup" aria-label="Data source">
              <ToggleGroup
                type="single"
                value={form.inputMode}
                onValueChange={(v) => {
                  if (v) form.setInputMode(v as 'dataset' | 'text')
                }}
              >
                <ToggleGroupItem
                  value="dataset"
                  className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
                >
                  Use a dataset
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="text"
                  className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
                >
                  Paste text
                </ToggleGroupItem>
              </ToggleGroup>
            </div>

            {form.inputMode === 'text' && (
              <div className="relative">
                <textarea
                  value={form.textInput}
                  onChange={(e) => form.setTextInput(e.target.value)}
                  placeholder="Paste any text to train on — stories, docs, conversations, code..."
                  rows={4}
                  className="w-full rounded-md border border-border/60 bg-background p-3 pb-7 text-[11px] font-mono text-foreground resize-y min-h-[80px]"
                  aria-label="Training text input"
                />
                <span
                  className="absolute bottom-1.5 right-2 text-[10px] text-muted-foreground/50 tabular-nums"
                  aria-live="polite"
                  aria-atomic="true"
                >
                  {form.textInput.length > 0
                    ? `${form.textInput.length.toLocaleString()} chars · ~${Math.ceil(form.textInput.length / 4)} tokens`
                    : ''}
                </span>
              </div>
            )}

            {form.method === 'finetune' && form.availableModels.length > 0 && (
              <div className="flex flex-col gap-1.5">
                <label
                  htmlFor="train-base-model"
                  className="text-[10px] text-muted-foreground uppercase tracking-wider"
                >
                  Base model
                </label>
                <Select value={form.selectedModel} onValueChange={form.setSelectedModel}>
                  <SelectTrigger
                    id="train-base-model"
                    className="h-8 text-xs font-mono max-w-sm"
                    aria-label="Base model"
                  >
                    <SelectValue placeholder="Select model..." />
                  </SelectTrigger>
                  <SelectContent>
                    {form.availableModels.map((id) => (
                      <SelectItem key={id} value={id}>
                        {id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Expert controls: default-closed fold (Phase 3 sign-off 2026-10-09) */}
            <FoldSection heading="Show advanced options">
              <div className="space-y-3">
                {/* Tokenizer selection */}
                {form.method !== 'finetune' && (
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] text-muted-foreground uppercase tracking-wider">
                      Tokenizer
                    </label>
                    <div
                      className="flex items-center gap-1"
                      role="radiogroup"
                      aria-label="Tokenizer algorithm"
                    >
                      <ToggleGroup
                        type="single"
                        value={form.algo}
                        onValueChange={(v) => {
                          if (v && typeof v === 'string') form.setAlgo(v)
                        }}
                      >
                        <ToggleGroupItem
                          value="bpe"
                          className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
                        >
                          BPE
                        </ToggleGroupItem>
                        <ToggleGroupItem
                          value="unigram"
                          className="px-3 py-1 rounded-md text-[11px] data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:font-medium"
                        >
                          Unigram
                        </ToggleGroupItem>
                      </ToggleGroup>
                    </div>
                    <p className="text-[10px] text-muted-foreground/60">
                      {form.algo === 'bpe' &&
                        'Byte Pair Encoding — fast, deterministic tokenization. Best for most use cases.'}
                      {form.algo === 'unigram' &&
                        'Unigram — probabilistic segmentation with subword regularization. Better for noisy data.'}
                    </p>
                  </div>
                )}

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <div className="flex flex-col gap-1">
                    <Label htmlFor="train-epochs" variant="uppercase">
                      Epochs
                    </Label>
                    <Input
                      id="train-epochs"
                      type="number"
                      min={TRAINING_LIMITS.epochs.min}
                      max={TRAINING_LIMITS.epochs.max}
                      value={epochsDraft}
                      onChange={(e) => setEpochsDraft(e.target.value)}
                      onBlur={() =>
                        commitField(
                          'epochs',
                          epochsDraft,
                          form.trainingEpochs,
                          setEpochsDraft,
                          form.setTrainingEpochs,
                        )
                      }
                      error={hp.epochs !== null}
                      aria-invalid={hp.epochs !== null}
                      aria-describedby={hpErrors.length > 0 ? 'configure-hp-errors' : undefined}
                      className="h-7 text-[11px] font-mono"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label htmlFor="train-batch" variant="uppercase">
                      Batch size
                    </Label>
                    <Input
                      id="train-batch"
                      type="number"
                      min={TRAINING_LIMITS.batch_size.min}
                      max={TRAINING_LIMITS.batch_size.max}
                      value={batchDraft}
                      onChange={(e) => setBatchDraft(e.target.value)}
                      onBlur={() =>
                        commitField(
                          'batch_size',
                          batchDraft,
                          form.trainingBatchSize,
                          setBatchDraft,
                          form.setTrainingBatchSize,
                        )
                      }
                      error={hp.batch !== null}
                      aria-invalid={hp.batch !== null}
                      aria-describedby={hpErrors.length > 0 ? 'configure-hp-errors' : undefined}
                      className="h-7 text-[11px] font-mono"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label htmlFor="train-lr" variant="uppercase">
                      Learning rate
                    </Label>
                    <Input
                      id="train-lr"
                      type="text"
                      inputMode="decimal"
                      value={lrDraft}
                      onChange={(e) => setLrDraft(e.target.value)}
                      onBlur={() =>
                        commitField(
                          'learning_rate',
                          lrDraft,
                          form.trainingLR,
                          setLrDraft,
                          form.setTrainingLR,
                        )
                      }
                      error={hp.lr !== null}
                      aria-invalid={hp.lr !== null}
                      aria-describedby={hpErrors.length > 0 ? 'configure-hp-errors' : undefined}
                      className="h-7 text-[11px] font-mono"
                    />
                  </div>
                </div>
                {form.method === 'finetune' && (
                  <div className="space-y-1.5">
                    <label className="flex items-center gap-2 text-[11px]">
                      <Checkbox
                        checked={form.useLoRA}
                        onCheckedChange={(e) => form.setUseLoRA(e === true)}
                        className="rounded border-border"
                        aria-label="Enable LoRA"
                      />
                      Use LoRA (parameter-efficient fine-tuning)
                    </label>
                    {form.useLoRA && (
                      <div className="grid grid-cols-2 gap-2 pl-5">
                        <div className="flex flex-col gap-1">
                          <Label htmlFor="lora-rank" variant="uppercase">
                            Rank
                          </Label>
                          <Input
                            id="lora-rank"
                            type="number"
                            min={1}
                            max={128}
                            value={form.loraRank}
                            onChange={(e) => form.setLoraRank(Number(e.target.value) || 8)}
                            className="h-7 text-[11px] font-mono"
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <Label htmlFor="lora-alpha" variant="uppercase">
                            Alpha
                          </Label>
                          <Input
                            id="lora-alpha"
                            type="number"
                            min={1}
                            max={256}
                            value={form.loraAlpha}
                            onChange={(e) => form.setLoraAlpha(Number(e.target.value) || 16)}
                            className="h-7 text-[11px] font-mono"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
                {checkpoints && checkpoints.checkpoints.length > 0 && (
                  <div className="flex flex-col gap-1.5">
                    <label
                      htmlFor="resume-checkpoint"
                      className="text-[10px] text-muted-foreground uppercase tracking-wider"
                    >
                      Resume from checkpoint (optional)
                    </label>
                    <Select value={form.resumeCheckpoint} onValueChange={form.setResumeCheckpoint}>
                      <SelectTrigger
                        id="resume-checkpoint"
                        className="h-8 text-xs font-mono"
                        aria-label="Resume from checkpoint"
                      >
                        <SelectValue placeholder="Start fresh" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="">Start fresh</SelectItem>
                        {checkpoints.checkpoints.map((c) => (
                          <SelectItem key={c.name} value={c.name}>
                            {c.name} {c.loss != null ? `(${c.loss.toFixed(4)})` : ''}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </FoldSection>
            {/* HP validation stays visible even when the fold is collapsed — it gates Next */}
            {hpErrors.length > 0 && (
              <div id="configure-hp-errors" className="text-[11px] text-destructive space-y-0.5">
                {hpErrors.map((e) => (
                  <div key={e}>{e}</div>
                ))}
              </div>
            )}

            {/* Training recommendation */}
            {recommendation && (
              <div className="rounded-lg border border-primary/20 bg-primary/5 p-2.5 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-medium text-primary">
                    Recommended Configuration
                  </span>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px]"
                    onClick={applyRecommendation}
                  >
                    Apply
                  </Button>
                </div>
                <div className="text-[10px] text-muted-foreground/60 space-y-0.5">
                  <div>Dataset: {recommendation.dataset_size.toLocaleString()} examples</div>
                  <div>Reason: {recommendation.recommendation.reason}</div>
                  <div className="flex gap-3 tabular-nums">
                    <span>LR: {recommendation.recommendation.learning_rate.toExponential(1)}</span>
                    <span>Batch: {recommendation.recommendation.batch_size}</span>
                    <span>Epochs: {recommendation.recommendation.epochs}</span>
                  </div>
                  {recommendation.tips.length > 0 && (
                    <div className="pt-1 border-t border-border/30">{recommendation.tips[0]}</div>
                  )}
                </div>
              </div>
            )}
            {loadingRec && (
              <div className="text-[10px] text-muted-foreground animate-pulse">
                Loading recommendation...
              </div>
            )}
          </>
        ) : (
          /* Vision training — no data source or hyperparams needed */
          <div className="text-sm text-muted-foreground py-2">
            Vision training uses the selected dataset to teach image understanding.
          </div>
        )}

        <div className="flex items-center gap-1.5 pt-1">
          <Button size="sm" className="h-7 text-[11px]" disabled={!canAdvance} onClick={onNext}>
            Next: Train
          </Button>
          <Button size="sm" variant="ghost" className="h-7 text-[11px]" onClick={onBack}>
            Back
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
