# sloughGPT — User Flows

Complete mapping of every user journey through the web UI.

---

## Table of Contents

1. [First Launch & Onboarding](#1-first-launch--onboarding)
2. [Authentication](#2-authentication)
3. [Chat](#3-chat)
4. [Model Management](#4-model-management)
5. [Personalities & Souls](#5-personalities--souls)
6. [Training](#6-training)
7. [Auto Training](#7-auto-training)
8. [Datasets](#8-datasets)
9. [Knowledge & Memory](#9-knowledge--memory)
10. [Agents](#10-agents)
11. [Monitoring & Observability](#11-monitoring--observability)
12. [Settings & Configuration](#12-settings--configuration)
13. [Feedback & Improvement](#13-feedback--improvement)
14. [Tokenizer & Embeddings](#14-tokenizer--embeddings)
15. [Benchmarking & Experiments](#15-benchmarking--experiments)
16. [Error Recovery Flows](#16-error-recovery-flows)
17. [Keyboard Shortcuts Reference](#17-keyboard-shortcuts-reference)

---

## 1. First Launch & Onboarding

### Flow: Cold Start → Ready to Use

```
User opens app
  → Root layout loads (fonts, theme, providers)
  → AppLayout mounts (sidebar, status bar, SSE connection)
  → Homepage renders
    → useLiveStatus starts SSE stream
    → useHomePageData fetches sessions, training, knowledge, feedback, datasets
    → Model readiness polling begins (every 3s)
```

**States the user sees:**

| Phase | What renders | Duration |
|-------|-------------|----------|
| Connecting | "Connecting..." subtitle, skeleton cards | 1-5s |
| Starting up | Progress bar (step/total), startup message | 30-90s |
| Online, no model | "No model loaded" card, "Load a model" button | Until model loaded |
| Online, model loaded | Active model banner, stats grid, quick actions | Steady state |

**Onboarding Card:**
- Appears on first visit (stored in IndexedDB: `onboarding_dismissed`)
- Shows welcome message with quick-start steps
- "Skip" button dismisses permanently
- Re-appears if user clears browser data

**User actions from homepage:**
- Click "Switch" → `/models` (load a model)
- Click "Chat" → `/chat` (start conversation)
- Click "Start training" → `/training` (if datasets exist)
- Click nav grid cards → respective pages

---

## 2. Authentication

### Flow: Login

```
/auth
  → Login form (username + password)
  → Submit → POST /auth/login
  → Response: { token, user }
  → Token stored in IndexedDB (chatDB.setKV('auth_token', token))
  → Redirect to /
```

### Flow: Register

```
/auth
  → Toggle to "Register" mode
  → Form: username + email + password
  → Submit → POST /auth/register
  → Auto-login after registration
  → Token stored in IndexedDB
  → Redirect to /
```

### Flow: Token Management

```
/auth
  → "Verify Token" button → validates current token
  → Shows session info (user, expiry)
  → "Logout" → clears token from IndexedDB, resets state
```

### Flow: Unauthenticated Access

```
User visits any / page without token
  → SessionProvider shows loading state
  → If no valid session → redirect to /auth
  → After login → redirect back to original page
```

---

## 3. Chat

### Flow: Start New Conversation

```
/chat (or Ctrl+N)
  → ChatProvider mounts
  → Empty conversation state
  → User types message in PromptComposer
  → Submit (Enter)
    → chatController.send(message)
    → Streaming response via SSE
    → Messages render in ChatThread
    → Conversation saved to backend
    → Sidebar updates with new conversation
```

### Flow: Continue Existing Conversation

```
/sidebar click on conversation
  → /chat?session=<id>
  → ChatProvider loads session messages
  → Messages render in ChatThread
  → User can send new messages
  → Context maintained across messages
```

### Flow: File Attachment

```
Chat interface
  → Click attachment button or drag-drop file
  → File uploaded to backend
  → AttachmentChip appears in prompt
  → File content included in message context
  → Supports: images (vision), text files, code files
```

### Flow: Slash Commands

```
Chat input
  → Type "/"
  → Command menu appears (autocomplete)
  → Select command (e.g., /clear, /model, /soul)
  → Command executes
  → Some commands show dialogs (e.g., /model opens model picker)
```

### Flow: Model Switching in Chat

```
Chat toolbar → Model selector dropdown
  → Lists available models
  → Select new model
  → Model switches for current conversation
  → Subsequent messages use new model
```

### Flow: Voice Input

```
Chat interface
  → Click microphone icon
  → Browser requests microphone permission
  → Voice recorded → transcribed
  → Transcription appears in input
  → User sends transcribed text
```

### Flow: Vision Studio

```
Chat dialog → Vision Studio
  → Upload image
  → Select analysis mode
  → AI analyzes image
  → Results displayed in workspace
  → Can save/attach results to conversation
```

### Flow: Search Conversations

```
Chat sidebar → Search (or Ctrl+Shift+F)
  → Type search query
  → Matching conversations highlighted
  → Click result → navigate to conversation
```

### Flow: Settings Panel (Per-Chat)

```
Chat toolbar → Settings toggle
  → Settings panel slides in
  → Adjust: temperature, max tokens, streaming toggle
  → Changes apply to current conversation
  → Persisted per-conversation
```

---

## 4. Model Management

### Flow: View Available Models

```
/models
  → Model catalog loads
  → Shows: model name, size, status, quantization
  → Current model highlighted
  → Can sort/filter by size, type, status
```

### Flow: Load/Unload Model

```
/models
  → Click "Load" on a model
  → Model loading indicator appears
  → SSE stream shows loading progress
  → On success: model marked as "loaded"
  → Homepage shows active model banner
  → Can now chat with that model
```

### Flow: Switch Model

```
/models
  → Click "Switch" on different model
  → Current model unloads
  → New model loads
  → All conversations now use new model
  → Previous conversations retain their model reference
```

### Flow: Model Details

```
/models → Click model card
  → /model/[id]
  → Shows: architecture, parameters, quantization
  → Checkpoint list
  → Performance metrics
  → Can delete model
```

### Flow: Checkpoint Management

```
/models → Checkpoints section
  → List of checkpoints for current model
  → Can switch between checkpoints
  → Can export checkpoint
  → Can delete checkpoint
```

---

## 5. Personalities & Souls

### Flow: View Personalities

```
/souls
  → List of available personalities (souls)
  → Each soul shows: name, description, traits
  → Current active soul highlighted
```

### Flow: Switch Personality

```
/souls
  → Click "Activate" on a soul
  → Personality changes globally
  → Chat responses now use new personality
  → Homepage shows current personality
```

### Flow: Edit Trait Weights

```
/souls → Click soul → PersonalityProfileCard
  → Slider for each trait (creativity, formality, etc.)
  → Adjust weights
  → Preview changes
  → Save
```

### Flow: Create Custom Soul

```
/souls → "Create" button
  → Form: name, description, trait weights
  → Save → new soul added to list
  → Can activate immediately
```

### Flow: Personality Analytics

```
/souls → Analytics section
  → Shows which souls used most
  → Response quality per personality
  → User satisfaction scores
```

---

## 6. Training

### Flow: Start Training

```
/training
  → Training page loads
  → DatasetSelector shows available datasets
  → Select dataset
  → Configure hyperparameters:
    - Epochs (slider)
    - Batch size (slider)
    - Learning rate (slider)
    - LoRA rank (slider)
  → Click "Start Training" (or Ctrl+Enter)
  → Training job created
  → Live log streaming begins
  → Loss curve updates in real-time
```

### Flow: Monitor Training

```
/training (during training)
  → TrainingLogCard shows live logs
  → LossCurve shows training progress
  → Progress bar shows completion %
  → Browser notification on completion
  → Can stop training early
```

### Flow: Test Trained Model

```
/training → Results tab
  → "Test Model" button (or Ctrl+Shift+T)
  → TestModelDialog opens
  → Enter test prompt
  → Model responds with trained behavior
  → Can iterate and test multiple prompts
```

### Flow: View Training History

```
/training → History tab
  → List of past training jobs
  → Each shows: name, status, duration, metrics
  → Click job → /training/job/[id]
  → Detailed view: logs, checkpoints, metrics
```

### Flow: Export Training Metrics

```
/training → Results tab
  → "Export" button
  → Downloads JSON with:
    - Loss curves
    - Hyperparameters
    - Checkpoint paths
    - Duration
    - Final metrics
```

### Flow: Individual Job Detail

```
/training/job/[id]
  → Full training log
  → Checkpoint list with download links
  → Loss curve visualization
  → Hyperparameter summary
  → Can restart from checkpoint
```

---

## 7. Auto Training

### Flow: Auto Training Workflow

```
/auto-train
  → Automated pipeline view
  → Select dataset
  → Configure pipeline stages
  → Start auto-training
  → Pipeline progresses through stages:
    1. Data preparation
    2. Tokenization
    3. Training
    4. Evaluation
    5. Checkpoint selection
  → Results displayed with loss charts
```

---

## 8. Datasets

### Flow: Import Dataset

```
/datasets
  → Click "Import" (or N key)
  → DatasetImportDialog opens
  → Select source:
    - File upload (JSON, CSV, TXT)
    - Paste text
    - From knowledge base
  → Preview data
  → Name dataset
  → Import
  → Dataset appears in list
```

### Flow: Browse Datasets

```
/datasets
  → List of all datasets
  → Shows: name, size, samples, date
  → Search by name
  → Sort by date, size, samples
  → Click dataset → /dataset/[id]
```

### Flow: Dataset Detail

```
/dataset/[id]
  → Full dataset view
  → Sample preview (first N rows)
  → Statistics: total samples, size, format
  → Can edit metadata
  → Can delete dataset
  → "Train with this" button → /training?dataset=<id>
```

### Flow: Compare Datasets

```
/datasets
  → Select multiple datasets (checkboxes)
  → "Compare" button
  → Side-by-side view
  → Shows: size difference, sample overlap, format differences
```

### Flow: Quick Train from Dataset

```
/datasets → Click play icon on dataset
  → Redirects to /training?dataset=<id>
  → Dataset pre-selected
  → User configures hyperparameters
  → Starts training
```

### Flow: Delete Dataset

```
/datasets → Click delete on dataset
  → ConfirmDialog appears
  → Confirm deletion
  → Dataset removed from list
  → Backend cleanup
```

---

## 9. Knowledge & Memory

### Flow: Add Knowledge

```
/knowledge
  → "Add Fact" button
  → Form: content textarea + topic input
  → Topic auto-suggests from existing topics
  → Submit
  → Fact added to knowledge base
  → Stats updated
```

### Flow: Import Knowledge

```
/knowledge → "Import" button
  → File upload dialog
  → Supports: JSON, CSV, TXT
  → Preview imported facts
  → Confirm import
  → Facts added to knowledge base
```

### Flow: Search Knowledge

```
/knowledge → Search bar
  → Type query
  → Semantic search across all facts
  → Results ranked by relevance
  → Can filter by topic
```

### Flow: Edit Knowledge

```
/knowledge → Click fact
  → Inline edit mode
  → Modify content, topic, importance
  → Save changes
```

### Flow: Bulk Operations

```
/knowledge → Select multiple facts (checkboxes)
  → Bulk actions:
    - Delete selected
    - Move to topic
    - Export selected
    - Change importance
  → Confirm action
  → Batch processing
```

### Flow: Train Memory Adapter

```
/knowledge → "Train Memory" button
  → Select knowledge subset
  → Configure adapter settings
  → Start training
  → LoRA adapter created
  → Can use in chat for knowledge-aware responses
```

### Flow: Deep Memory (RAG)

```
/knowledge → Deep Memory section
  → Sync knowledge to vector store
  → Configure retrieval settings
  → Enable RAG mode
  → Chat responses now include retrieved knowledge
```

### Flow: Spaced Review

```
/knowledge → Review section
  → Facts due for review displayed
  → User rates retention
  → Schedule adjusted based on performance
  → Long-term memory optimization
```

---

## 10. Agents

### Flow: Create Agent

```
/agents → "Create" button
  → Form:
    - Name
    - Description
    - System instructions
    - Tool selection (from templates or custom)
  → Save
  → Agent appears in list
```

### Flow: Execute Agent

```
/agents → Click agent → Execute
  → Prompt input
  → Submit
  → Agent processes with assigned tools
  → Streaming response
  → Tool calls shown inline
  → Results displayed
```

### Flow: Multi-Agent Orchestration

```
/agents → Orchestrate tab
  → Enter goal + context
  → System decomposes into tasks
  → Tasks assigned to appropriate agents
  → Real-time task status
  → Results aggregated
  → Final response compiled
```

### Flow: Agent Templates

```
/agents → Templates section
  → Pre-built agent templates:
    - Research agent
    - Code agent
    - Writing agent
    - Analysis agent
  → Click template → creates agent from template
  → Customize after creation
```

### Flow: Run History

```
/agents → History tab
  → List of past agent executions
  → Each shows: agent, prompt, duration, status
  → Click run → full execution details
  → Can re-run with same prompt
```

---

## 11. Monitoring & Observability

### Flow: System Health Dashboard

```
/monitoring
  → Real-time metrics display
  → Auto-refresh (configurable interval)
  → Metrics shown:
    - CPU usage (%)
    - Memory usage (%)
    - Disk usage
    - GPU usage (if available)
    - Request count
    - Error rate
    - Latency (p50, p95, p99)
    - Uptime
```

### Flow: Diagnostics

```
/monitoring → Diagnostics section
  → Expand fold section
  → Shows:
    - Inference pool status
    - Executor pool status
    - KV cache stats
    - DPO training status
  → Can trigger diagnostic tests
```

### Flow: Alerts

```
/monitoring → Alerts section
  → Configurable thresholds:
    - CPU > 90% → warning
    - Memory > 85% → warning
    - Error rate > 5% → critical
  → Visual alerts with color coding
  → Can dismiss/acknowledge alerts
```

### Flow: Export System Report

```
/monitoring → "Export" button
  → Generates JSON report:
    - All current metrics
    - Recent errors
    - Training status
    - System info
  → Downloads as file
```

### Flow: Client Error Monitoring

```
/errors
  → Grouped errors by type
  → Recent errors list
  → Hourly trend chart
  → SSE live stream of new errors
  → Can search/filter errors
  → Export errors as JSON
  → Clear error history
```

### Flow: Security Audit Logs

```
/security
  → Audit log entries
  → Shows: action, user, timestamp, IP
  → Filter by action type
  → Load older logs
  → API key status display
```

---

## 12. Settings & Configuration

### Flow: Appearance Settings

```
/settings → Appearance section
  → Theme: Light / Dark / System
  → Palette: Color scheme selection
  → Changes apply immediately
  → Persisted to backend
```

### Flow: Language Settings

```
/settings → Language section
  → Select language from dropdown
  → UI text updates immediately
  → Supported: English, and others
```

### Flow: Connection Settings

```
/settings → Connection section
  → API URL input
  → HuggingFace token input
  → "Test Connection" button
  → Shows latency on success
  → Error message on failure
```

### Flow: Chat Defaults

```
/settings → Chat section
  → Temperature slider (0-2)
  → Max tokens slider
  → Top-p slider
  → Top-k slider
  → Streaming toggle
  → Changes apply to new conversations
```

### Flow: Custom Instructions

```
/settings → Memory section
  → System instructions textarea
  → These instructions included in all chat contexts
  → Save → applies to future conversations
```

### Flow: Backup & Restore

```
/settings → Backup section
  → "Export Settings" → downloads JSON
  → "Import Settings" → file upload
  → Validates imported settings
  → Applies imported settings
```

### Flow: Danger Zone

```
/settings → Danger Zone section
  → "Clear Chat History" → ConfirmDialog → deletes all conversations
  → "Reset Settings" → ConfirmDialog → restores defaults
  → "Delete All Data" → ConfirmDialog → nuclear option
```

---

## 13. Feedback & Improvement

### Flow: Give Feedback

```
/chat → After AI response
  → Thumbs up/down buttons
  → Click → feedback recorded
  → Optional: add text feedback
  → Feedback stored in backend
```

### Flow: View Feedback Analytics

```
/feedback
  → Feedback stats:
    - Total feedback count
    - Thumbs up / thumbs down ratio
    - Per-model breakdown
    - Trend over time
  → Conversation list with feedback
```

### Flow: Train from Feedback

```
/feedback → "Train from feedback" button
  → Select feedback subset
  → Configure training
  → Start DPO training
  → Model learns from user preferences
```

### Flow: Feedback Workflow

```
/feedback → Workflow section
  → Aggregate: combines feedback into training data
  → Prune: removes low-quality feedback
  → Export: downloads feedback as JSON
  → Status indicators for each operation
```

---

## 14. Tokenizer & Embeddings

### Flow: Tokenize Text

```
/tokenizer
  → Input text field
  → Submit → text tokenized
  → Shows: token count, token IDs, decoded tokens
  → Visual token breakdown
```

### Flow: Train Tokenizer

```
/tokenizer → "Train" tab
  → Select training data
  → Configure vocab size
  → Train new tokenizer
  → Compare with existing tokenizers
```

### Flow: Token Tree Visualization

```
/token-tree
  → Visual tree of token merges
  → Click nodes to see merge operations
  → Path visualization from text to tokens
  → Compare different tokenizers
```

### Flow: Vector Store

```
/vector
  → Initialize vector store (ChromaDB or in-memory)
  → Add documents
  → Semantic search
  → Manage vector collections
  → Integration with knowledge base
```

---

## 15. Benchmarking & Experiments

### Flow: Run Benchmark

```
/benchmark
  → Select models to compare
  → Choose benchmark suite
  → Run benchmark
  → Results:
    - Throughput (tokens/sec)
    - Latency (p50, p95, p99)
    - Quality scores
    - Perplexity
  → Side-by-side comparison
```

### Flow: Experiment Tracking

```
/experiments
  → Create experiment
  → Log metrics and parameters
  → Mark experiment as complete
  → Export experiment data
  → Compare experiments
```

### Flow: LoRA Evaluation

```
/lora-eval
  → Select LoRA adapters
  → Run evaluation
  → Compare adapter performance
  → Quality metrics
  → Integration with model page
```

---

## 16. Error Recovery Flows

### Flow: API Offline

```
Homepage detects API offline
  → Shows "Server offline" card
  → Retries connection automatically
  → On reconnection → status updates
  → User can manually refresh
```

### Flow: Model Loading Failure

```
Model fails to load
  → Error toast appears
  → Model status shows "failed"
  → User can retry load
  → Check logs at /errors
```

### Flow: Training Failure

```
Training job fails
  → Job status → "failed"
  → Error message in training log
  → Checkpoints from partial training retained
  → User can:
    - Retry with different hyperparameters
    - Check /errors for details
    - Contact support with exported report
```

### Flow: Chat Error

```
Chat message fails to send
  → Error toast appears
  → Message marked as failed
  → User can retry sending
  → Partial responses preserved
```

### Flow: Network Disconnection

```
Network drops during session
  → SSE connection lost
  → Status bar shows "Reconnecting..."
  → Auto-reconnect with exponential backoff
  → On reconnection → state synced
  → Missed messages retrieved
```

### Flow: Session Recovery

```
Browser crash/reload
  → IndexedDB preserves:
    - Current conversation ID
    - Auth token
    - Onboarding state
  → On reload → session restored
  → Chat loads last conversation
```

---

## 17. Keyboard Shortcuts Reference

### Global

| Key | Action |
|-----|--------|
| `1` | Navigate to Chat |
| `2` | Navigate to Training |
| `3` | Navigate to Datasets |
| `4` | Navigate to Models |
| `5` | Navigate to Agents |
| `6` | Navigate to Souls |
| `7` | Navigate to Monitoring |
| `8` | Navigate to Knowledge |
| `Shift+A` | Open Settings |
| `Ctrl+K` | Command Palette |
| `Ctrl+\` | Toggle Sidebar |
| `?` | Show Keyboard Shortcuts |

### Chat

| Key | Action |
|-----|--------|
| `Ctrl+N` | New Conversation |
| `Ctrl+Shift+F` | Search Conversations |
| `Enter` | Send Message |
| `Shift+Enter` | New Line in Input |
| `Escape` | Cancel/Close Dialog |
| `/` | Open Slash Commands |
| `Ctrl+Shift+T` | Test Model |

### Training

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Start Training |
| `Ctrl+Shift+T` | Test Model |

### Datasets

| Key | Action |
|-----|--------|
| `N` | Import New Dataset |

---

## Cross-Page Navigation Map

```
                          ┌─────────────┐
                          │  Homepage   │
                          │     /       │
                          └──────┬──────┘
                 ┌───────────────┼───────────────┐
                 │               │               │
          ┌──────▼──────┐ ┌─────▼─────┐ ┌───────▼───────┐
          │   /models   │ │   /chat   │ │  /training    │
          │   Models    │ │   Chat    │ │  Training     │
          └──────┬──────┘ └─────┬─────┘ └───────┬───────┘
                 │               │               │
       ┌─────────┼─────────┐    │         ┌─────┼─────┐
       │         │         │    │         │           │
  ┌────▼───┐ ┌──▼───┐ ┌───▼┐  │    ┌────▼───┐ ┌────▼───┐
  │/souls  │ │/bench│ │/lora│  │    │/dataset│ │/auto   │
  │Souls   │ │Mark  │ │Eval │  │    │Detail  │ │Train   │
  └────────┘ └──────┘ └─────┘  │    └────────┘ └────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              ┌─────▼────┐ ┌───▼────┐ ┌────▼─────┐
              │/knowledge│ │/feedback│ │/monitor  │
              │Knowledge │ │Feedback│ │Monitoring│
              └──────────┘ └────────┘ └──────────┘
```

---

## Data Flow Summary

### User Action → Backend → UI Update

| Action | API Call | UI Update |
|--------|----------|-----------|
| Send chat message | `POST /chat` | SSE stream → messages render |
| Load model | `POST /models/load` | SSE status → loading indicator → active banner |
| Start training | `POST /training` | SSE logs → live log card → loss curve |
| Add knowledge | `POST /knowledge` | Stats update → knowledge list |
| Import dataset | `POST /datasets/import` | Dataset list refreshes |
| Switch personality | `POST /souls/activate` | Chat responses change tone |
| Give feedback | `POST /feedback` | Stats update → feedback bar |
| Update settings | `PUT /settings` | Immediate UI re-render |

### State Persistence

| Storage | Contents |
|---------|----------|
| IndexedDB (`chatDB`) | Auth token, current session, onboarding state |
| Backend API | All data (conversations, models, datasets, knowledge) |
| URL params | Dataset selection, session selection |
| Local state | UI state (panels open, scroll position) |
| Zustand stores | Toasts, API monitor, settings, model readiness |

---

*Last updated: 2026-09-03*
