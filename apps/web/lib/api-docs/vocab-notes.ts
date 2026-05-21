/** Shared doc fragments for endpoints tied to trainer checkpoints / char-LM eval. */

/** Native trainer step_*.pt on the server includes char vocab for eval; see repo CONTRIBUTING.md. */
export const TRAINING_CHECKPOINT_VOCAB_NOTE =
  ' Trainer step_*.pt files include stoi/itos/chars (char-LM eval). See docs/policies/CONTRIBUTING.md (Checkpoint vocabulary).'

/** Clarifies resolve has no artifacts; training endpoints write charset-aware step_*.pt. */
export const TRAIN_RESOLVE_VOCAB_NOTE =
  ' Does not write checkpoints. After POST /train or POST /training/start, native step_*.pt embeds the same char vocab; see docs/policies/CONTRIBUTING.md (Checkpoint vocabulary).'

/** Optional checkpoint path on completed jobs uses native trainer step_*.pt charset semantics. */
export const TRAINING_JOBS_VOCAB_NOTE =
  ' Jobs may include a checkpoint path; native step_*.pt embeds stoi/itos/chars for cli.py eval. See docs/policies/CONTRIBUTING.md (Checkpoint vocabulary).'

/** Deployment export vs native char-LM checkpoint vocab (cli.py eval). */
export const MODEL_EXPORT_VOCAB_NOTE =
  ' Exported artifacts do not match trainer step_*.pt charset semantics for char-LM perplexity; see docs/policies/CONTRIBUTING.md (Checkpoint vocabulary).'

/** API perplexity uses loaded model+tokenizer; char-LM file eval is cli.py eval / lm_eval_char. */
export const BENCHMARK_PERPLEXITY_VOCAB_NOTE =
  ' Distinct from char-LM cli.py eval on a checkpoint file (stoi/itos from step_*.pt); see docs/policies/CONTRIBUTING.md (Checkpoint vocabulary).'
