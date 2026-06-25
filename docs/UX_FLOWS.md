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
