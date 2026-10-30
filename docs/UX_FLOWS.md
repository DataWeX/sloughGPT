# User Experience Flows

Plain-English guide for building user-facing features. No backend jargon, no
technical implementation details — only what the user sees, clicks, and feels.

---

## Persona

This app is for **everyday people** who want AI help in daily life. They are
not developers, not ML engineers, not power users. They want things to **just
work** — one click, one tap, no setup, no confusing words.

**Golden rule:** If the UI needs an acronym, a model name, a parameter slider,
or a technical term — it's wrong.

---

**Implementation note (Sep 2026):** Tool flows below — Writing, Rewrite,
Translate, Explain, Brainstorm, Decide, Wellness, Talk Out Loud, Create Images —
are all implemented inside the chat `ModeBar` (one `chat?mode=<mode>`
deep-link per flow), **not** as standalone pages. Flows 1 and 12 still have
their own pages (`/chat`, `/training`).

**Executable registry (Sep 2026):** Every flow below is also an executable
spec-linked journey in `domain/journeys/flows.py` — each `Flow` carries its
exact heading from this document as its `spec`, and
`tests/test_ux_flows_library.py` fails the suite if a heading is renamed or
a flow loses its anchor. Run them with
`.venv/bin/python -m domain.journeys` (`--list` shows ids + spec links;
`--flow <id>` for one; the live sweep needs a running stack).
**Exception:** flow 14 is doc-only for now — the plugins journey has no
executable sweep yet. The gate checks registry→doc anchors, so an
unregistered heading is allowed; register it when the sweep exists.

---

## Features

### 1. Chat That Remembers Me

**What the user sees:**

- Open the app → immediately start typing or talking
- The AI knows their name, their preferences, past conversations
- No "New chat" button that resets everything — memory is persistent
- A sidebar shows recent conversations by topic ("Drafting an email to mom",
  "Planning weekend trip", "Learning about climate change")

**User flow:**

1. Open app → typing cursor already blinking in input
2. Type "What was that recipe we talked about yesterday?"
3. AI remembers and answers with the recipe from yesterday's chat
4. User can scroll up through past conversations like a timeline

**What they should NOT see:**

- Model names (GPT-2, Qwen, etc.)
- Sliders for temperature, top-p, etc.
- Session IDs or chat IDs
- "New chat" as a primary action

---

### 2. Writing Assistant

**What the user sees:**

- A big text area labeled "What do you want to write?"
- Tone buttons: Friendly | Professional | Funny | Short | Detailed
- Type buttons: Email | Social Post | Story | Poem | Letter | Note
- A "Make it better" button that polishes what they've written

**User flow:**

1. Select tone (e.g. "Friendly") and type (e.g. "Email")
2. Type or dictate: "Tell my landlord the sink is broken and ask when he can fix it"
3. Tap "Write" → AI generates the email
4. Tap "Rewrite" → AI tries a different version
5. Tap "Copy" → ready to paste into Gmail/Outlook
6. Tap "Make shorter" or "Make funnier" → instant revision

**What they should NOT see:**

- Inference settings
- Model selection
- Token counts
- "Generate" — use "Write" or "Create" instead

---

### 3. Read My Files

**What the user sees:**

- A drop zone: "Drop a file here or click to upload"
- Supported: PDF, Word, text files
- After upload: a chat appears below the file, ready for questions
- Suggested questions appear automatically: "Summarize this", "What are the
  key points?", "Explain this section in simple terms"

**User flow:**

1. Drop a PDF (e.g. a lease agreement, a school article, a report from work)
2. AI says "Got it! I've read 15 pages. What do you want to know?"
3. Type "What's the move-out notice period?" → AI answers from the file
4. Type "Summarize this in 3 bullet points" → instant summary
5. All questions and answers stay in the chat for that file

**What they should NOT see:**

- "Knowledge base", "vector store", "ingestion", "indexing"
- "RAG", "embeddings", "chunks"
- Any loading bar that says "Indexing..." — just say "Reading your file..."
- File size limits (just handle it silently)

---

### 4. Brainstorm With Me

**What the user sees:**

- An open friendly input: "Let's think together. What's on your mind?"
- A few suggestion chips: "Name ideas", "Weekend plans", "Gift ideas",
  "Solve a problem", "Plan an event"
- Responses come back as ideas, not essays — bullet points, mind maps in text

**User flow:**

1. Type "I need gift ideas for my dad's 60th birthday. He loves fishing and cooking"
2. AI responds with 10 ideas in a friendly list
3. User says "I like #3 and #7, tell me more"
4. AI drills deeper on those two ideas
5. User can save the best ideas to a "notepad" for later

**What they should NOT see:**

- Any reference to "creative" or "deterministic" modes
- Temperature or creativity sliders
- Word limits

---

### 5. Rewrite & Polish

**What the user sees:**

- A text box labelled "Paste what you wrote"
- Buttons: Fix spelling/grammar | Make it shorter | Make it friendlier |
  Make it professional | Make it sound like me
- A side-by-side view: original on left, rewritten on right

**User flow:**

1. Paste a draft email or message
2. Tap "Make it professional" → polished version appears on the right
3. Tap "Fix grammar" → only grammar corrections highlighted
4. Tap "Make it friendlier" → casual version
5. Tap "Use this version" → replaces the original

**What they should NOT see:**

- Any reference to language models or AI settings
- "Repetition penalty" or "token limit" or other technical controls

---

### 6. Create Images

**What the user sees:**

- A text input: "Describe the image you want to create"
- A simple style picker (optional): Realistic | Cartoon | Watercolor |
  Sketch | Fantasy
- A "Create" button (not "Generate")
- After creation: Save | Share | Create another variation

**User flow:**

1. Type "A cozy cabin in the mountains at sunset, smoke coming from the chimney"
2. Pick a style (or leave as default)
3. Tap "Create" → image appears in a few seconds
4. Tap "Make another" → different version
5. Tap "Describe this in words" → AI writes a caption/prompt for the image
6. Tap "Save to my gallery" → images are stored in a personal gallery

**What they should NOT see:**

- "VLM", "diffusion", "model checkpoint", "training steps"
- Resolution or aspect ratio sliders (pick from presets if needed)
- Negative prompts or advanced options — keep it simple

---

### 7. Talk Out Loud

**What the user sees:**

- A microphone button that pulses when listening
- A speaker button to hear the AI's response read aloud
- When speaking: the AI transcribes in real-time (words appear as you speak)
- The AI responds with both text and voice

**User flow:**

1. Tap the microphone → start speaking
2. Words appear on screen as you speak
3. AI responds with text + reads it aloud
4. Continue the conversation naturally — speak again to respond
5. Tap the microphone again to stop listening
6. Transcript is saved in the conversation history

**What they should NOT see:**

- "Speech-to-text", "text-to-speech", "ASR", "TTS", "Web Speech API"
- Language selection (auto-detect)
- Volume or speed sliders (keep defaults good)

---

### 8. Translate

**What the user sees:**

- Two columns side by side — source language on left, target on right
- Auto-detect source language
- Big dropdown for target language (most common ones up top)
- Paste or type on left → translation appears instantly on right

**User flow:**

1. Type or paste "How much does this cost?" on the left
2. Select "Spanish" on the right → "¿Cuánto cuesta esto?" appears
3. Tap the speaker icon to hear the translation spoken
4. Tap "Copy translation" → ready to share
5. Tap "Swap languages" to reverse direction
6. Conversation mode: type back and forth, each side auto-translates

**What they should NOT see:**

- Any model selection or AI settings
- "Neural machine translation" or "translation model"

---

### 9. Help Me Decide

**What the user sees:**

- A simple prompt: "What are you deciding between?"
- Input fields for options: "Option A" and "Option B", with optional notes
- A "Help me decide" button
- Response: a pro/con table, a recommendation, and a simple explanation

**User flow:**

1. Type "Should I take the job in New York or stay in my current role?"
2. Add notes: "NY pays 20% more, current job has better hours"
3. Tap "Help me decide"
4. AI responds with a table: pros and cons for each, then a recommendation
5. User can ask "What if I only care about salary?" → AI re-evaluates
6. User can save the decision to their notepad

**What they should NOT see:**

- "Chain of thought", "reasoning", "analysis parameters"
- Any comparison of AI models or settings

---

### 10. Explain Things Simply

**What the user sees:**

- An input: "What do you want explained?"
- A difficulty selector: Simple (like explaining to a child) | Normal |
  Detailed
- A "Explain" button
- Response: plain language, examples, analogies

**User flow:**

1. Type "How does the internet work?"
2. Select "Simple" → AI explains using a post office analogy
3. User says "Tell me more about routers" → AI drills deeper
4. User can say "I'm still confused, try a different analogy"
5. The conversation keeps going until the user gets it

**What they should NOT see:**

- Any technical references to how the AI works
- "LLM", "neural network", "training data"

---

### 11. Make Me Well (Wellness)

**What the user sees:**

- A calming welcome screen
- Options: Sleep story | Meditation guide | Journal prompt |
  Breathing exercise | Positive affirmation
- No accounts, no tracking, no data stored (or at least feels private)

**User flow:**

1. Select "Sleep story" before bed
2. AI asks about preferences: "Do you want a story about the ocean, a forest,
   or a starry night?"
3. Picks "Ocean" → AI tells a gentle story with a calm voice
4. Story fades out after 5 minutes
5. User can set a timer: 10 min | 20 min | 30 min

**What they should NOT see:**

- "Voice synthesis model", "text generation", "personalization profile"
- Any clinical or medical claims — just "feeling better", "relaxing"

---

### 12. Train My AI

**What the user sees:**

- A simple page with one question: "What do you want to teach me?"
- A dataset selector showing their uploaded files and collections
- A big "Start Training" button
- While training: a friendly progress bar with plain-language status ("Learning from your data...")
- After training: a before/after comparison ("Here's how I talk now vs before")

**User flow:**

1. Open the training page
2. Pick a dataset from the list (or upload a new one — see "Read My Files")
3. Tap "Start Training" → training begins
4. Watch the loss chart go down (visual proof it's learning)
5. Training completes → "Your AI has been updated!" with a "Try it now" button
6. Tap "Try it now" → jumps to chat with the new model loaded
7. Optionally: see a before/after comparison of how the AI responds

**What they should NOT see:**

- Learning rate, batch size, epochs, LoRA rank, gradient accumulation
- "Configure training parameters" — defaults should work
- "RLHF", "GRPO", "KL coefficient", "warmup steps"
- Multiple training modes (turbo, session, feedback) — just one button
- Checkpoint filenames or paths
- "Hyperparameter grid search" or "model card" or "training analytics"

**Power user extras (hidden behind "Advanced" toggle):**

- Training quality slider (low / medium / high — maps to embed size + layers)
- Save/load training configs
- View detailed loss chart with step numbers
- Compare checkpoints side by side

---

### 13. Manage My Work (Kanban Board)

**What the user sees:**

- One board with their cards in simple columns: Inbox → Ready → Doing → Review → Verify → Done
- Cards can be moved forward or back; the board tells them right away if a move makes sense
- Blocked cards sit in their own lane so nothing gets lost
- A warning when a column is getting too full (WIP limit), in plain words
- A refresh button if the board looks stale

**User flow:**

1. Open the kanban page → the board loads with their cards in each column
2. Pick a card → see which columns it may legally move to
3. Move it (e.g. Ready → Doing) → the move is accepted or rejected with a short reason
4. If blocked: move it to Blocked, add context, unblock it when ready
5. Keep moving cards until everything sits in Done

**What they should NOT see:**

- Transition rule tables or engine internals (why a move is legal is enough)
- Hash chains, state signatures, or validation hashes
- Raw API errors — a rejected move gets a plain-language reason
- WIP limit numbers without context ("column full" beats "limit: 3")

---

### 14. Add an Ability (Plugins)

**What the user sees:**

- A "Plugins" list showing each installed ability with a plain-language
  name, one sentence about what it adds, and one switch
- Switching one on makes its panel appear right there in the list;
  switching it off makes it disappear — nothing else changes
- A "Look for new plugins" button for when they've added a plugin file

**User flow:**

1. Open Plugins → see "Site checks" with the switch off
2. Read one line: "Read-only checks across the app — looks at how it is
   doing and reports anything wrong"
3. Flip the switch on → the panel appears below with a "Run a check" button
4. Tap "Run a check" → results appear in the same panel, plain language
5. Flip the switch off → panel gone; the app works exactly as before

**What they should NOT see:**

- File names, Python, "plugin class", module paths, or load-error stack traces
- Lifecycle jargon ("setup", "teardown", "hooks", "registration")
- Anything broken when a plugin is off — off means gone, not error
- A plugin taking over the home screen or sidebar — abilities live in their
  panel

---

## Navigation & Layout (Plain English)

**Home screen (what they see first):**

- Simple greeting: "Hi [name], what can I help with?"
- A search bar that accepts anything — questions, tasks, files
- Below: quick action buttons in plain English:
  - 📝 Write something
  - 📄 Read a file
  - 🎨 Create an image
  - 💡 Brainstorm ideas
  - 🗣️ Talk instead

**Sidebar (simple, no tabs):**

- Recent conversations (by topic, not date)
- My files (uploaded documents)
- My gallery (saved images)
- Settings

**Settings (what a normal person would want):**

- Appearance: Light / Dark / Auto
- Voice: Choose the voice you like (friendly, calm, upbeat)
- Language: Which language to use
- Privacy: Clear my memory / Download my data
- About: What is this app?

---

## Design Principles

1. **No jargon** — If a word needs explaining, replace it.
2. **One click** — Every action should be one tap. No confirmation dialogs,
   no multi-step wizards.
3. **Silent defaults** — Everything should work with default settings.
   Advanced options are hidden unless explicitly requested.
4. **Remembers you** — The app should feel like it knows you. Name,
   preferences, history — no "Who are you?" every time.
5. **Forgiving** — Mistakes are undoable. No permanent actions without
   a safety net.
6. **Fast** — Show something immediately (even if it's a placeholder),
   then improve. Never a blank loading screen.
7. **Conversational** — Every feature is accessible through chat. If it
   can't be done by typing or talking, it's too complex.

---

## User Journeys

### Training: 3-Click Flow

**What the user sees:**

- A page titled "Teach me" with a simple subtitle: "Teach your agent from your data"
- No jargon — no "LoRA", "GRPO", "KL coefficient", "gradient accumulation"
- A 3-step pipeline: Pick data → Train → See results

**User flow:**

1. **Pick data** (click 1)
   - User sees their datasets as simple chips with names and sample counts
   - Can click a chip to select it, or use "Import" to add new data
   - Quick import from HuggingFace (one-click, no setup required)
   - Can paste text directly if no dataset exists
   - A preview shows a few lines so user knows what the data looks like
   - "Next" button advances to configure

2. **Configure & Train** (click 2)
   - Top: Text or Vision toggle (simple choice, no jargon)
   - Below: A "Recommended" banner appears automatically based on dataset size
     - Shows: "Based on your data, we recommend: LR 0.001, Batch 32, 10 epochs"
     - One "Apply" button to accept recommendations
   - Expert settings are hidden behind a "Show advanced options" toggle:
     - Only shows when user explicitly asks for it
     - Contains: epochs, batch size, learning rate, LoRA rank, tokenizer
   - "Start training" button (disabled until ready)

3. **See results** (click 3)
   - Live loss chart while training runs (visual: line going down = good)
   - Progress bar with ETA
   - When complete: "Training complete! Your AI now knows [dataset name]"
   - Checkpoint cards showing saved versions with loss and quality scores
   - "Load for chat" button to immediately use the trained model
   - "Train more" to go back and train again

**What they should NOT see:**

- Model names (GPT-2, Qwen, etc.) — just "Train from scratch" or "Continue training"
- Parameter sliders for temperature, top-p, etc.
- Session IDs, job IDs, or technical identifiers
- Jargon like "LoRA rank", "KL coefficient", "gradient accumulation"
- 15-parameter configuration forms
- Grid search, checkpoint comparison, or model card pages (power-user features)
- Webhooks, recovery cards, or API logs (system features)

**Progressive disclosure:**

- Default view: 3-step pipeline with recommended settings
- "Show advanced options" toggle reveals expert settings
- Results tab shows checkpoints and jobs (simple list, not analytics dashboard)
- No separate sub-pages for trends, compare, model-card, or grid-search

**Plain language translations:**

| Technical term  | What user sees                                  |
| --------------- | ----------------------------------------------- |
| Fine-tuning     | "Continue training"                             |
| Distillation    | "Train from scratch"                            |
| LoRA            | "Parameter-efficient mode" (hidden by default)  |
| Learning rate   | "Learning speed" (hidden by default)            |
| Batch size      | "Training batch size" (hidden by default)       |
| Checkpoint      | "Saved version"                                 |
| Loss            | "Learning progress" (visual chart, not numbers) |
| Epoch           | "Training round"                                |
| Hyperparameters | "Advanced options" (hidden by default)          |

### Planner Board — ux/kanban

**Persona:** infra-workbench user (operator), not the mainstream training
journey. The board is a work-tracking surface for everything running in the
project: what's moving, what's blocked, what's next.

**What the user sees:**

- One board: columns by state, cards with a title, priority, and tags
- Drag a card (or use the action bar) to move it between columns
- Immediate feedback when a move is accepted — or a clear reason when the
  engine rejects it ("blocked", "work in progress limit reached")
- An optional 3D view toggle for spatial overview; same board, same moves
- Refresh always re-reads the live board — no local ghost state

**User flow:**

1. **Open the board** (click 1) — see every card in its current column
2. **Move a card** (click 2) — drag or action bar; the engine validates the
   transition, then the card lands in its new column
3. **Understand a rejection** (click 3) — a rejected move says why in plain
   language, the card stays where it was

**What they should NOT see:**

- Raw transition codes, engine internals, or JSON payloads
- Silent failures — every rejected move produces visible feedback
- A different board state after refresh than the one they just acted on

---

## Mobile Journeys

Same golden rule as the web: plain language, no jargon, no technical terms.
These journeys are the in-code test paths for the mobile app — each one is
mapped to its implementation files and its tests in
`domain/journeys/flows.py`, and `scripts/journey_report.py` shows per-journey
results.

### Mobile: Organize My Conversations

**What the user sees:**

- Open the conversation drawer → recent chats, just like the web sidebar
- A "+ new folder" action creates a folder (e.g. "Work", "Trips") with a typed name
- Long-press a conversation → a familiar action sheet: Rename, Move to Folder, Clone, Delete
- Moving a chat into a folder shows a small colored badge next to the title
- Tapping a folder chip in the drawer filters the list to just that folder; tap again to see everything
- "Clone" makes an exact copy of a chat (same messages, new title "... (Copy)") so experimenting doesn't destroy the original
- Starred conversations are marked with a star and stay easy to find

**What they should NOT see:**

- Session IDs, storage keys, or cache messages
- "Duplicate session" or "deep copy" — it's just "Clone"
- Anything that loses their original chat when cloning

### Mobile: See What the AI Sees

**What the user sees:**

- A quiet "+ Context" nudge above the keyboard → tap, type a note (e.g. "I'm writing a wedding speech"), Send — from now on the AI knows it in this chat
- Details → "Context Window" shows what the AI is actually holding right now: system prompt, knowledge, the back-and-forth — grouped in plain-language sections
- Each section says roughly how big it is in plain words; a refresh button re-reads the live state
- If there's nothing yet, it says so plainly — never a blank screen or an error code

**What they should NOT see:**

- Token counts as raw jargon without context, model names, or prompt IDs
- API errors or stack traces when inspection fails — a friendly "Couldn't read it, try again"

### Mobile: Branch a Conversation

**What the user sees:**

- Long-press any message → a familiar action sheet with "Branch from here"
- Tapping it forks the chat: everything up to and including that message becomes a new conversation, opened right away
- The new chat is named "... (Branch)" and shows up in the drawer beside the original — the original is never changed
- Details → "Branch Lineage" shows where this chat came from: this session first, then its parents, in plain words
- Branching from a branch keeps the whole family tree reachable

**What they should NOT see:**

- Message IDs, parent pointers, or "fork" jargon — it's "Branch from here"
- A branch that arrives missing the earlier messages it should have copied
- Any change to the original conversation

### Mobile: Annotate a Message

**What the user sees:**

- Long-press a message → "Add note" → a small note box opens under it; type something only they care about ("the punchline goes here") → Save
- The note stays pinned under that message as a soft highlighted strip — visible while scrolling, quiet enough not to shout
- Long-press again → "Edit note" to change it, or clear the text and Save to remove it
- Notes survive closing and reopening the app, and even reloading the conversation — the note finds its message again
- Clone or branch a chat and each copy keeps its own notes namespace — annotating a copy never touches the original

**What they should NOT see:**

- Message IDs, storage keys, or anything that orphans a note when the chat reloads
- Notes leaking into clones/branches of the conversation
- A blank screen or error when the notes store is empty — no note means no strip
