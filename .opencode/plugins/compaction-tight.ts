// Session compaction prompt override for sloughGPT.
//
// Stock OpenCode template (compiled into the binary) mandates 6 sections,
// "keep every section even when empty", and a "Relevant Files" inventory.
// We want it tighter and workflow-aware instead: kanban card, branch, gates
// from our AGENTS.md SOP, no file list, no "(none)" scaffolding.
//
// Mechanism: `experimental.session.compacting` mutates the hook output in
// place (the runtime runs each hook as `hook(input, output)` and then returns
// `output` itself). Setting `prompt` replaces the built-in prompt wholesale
// (OpenCode then appends "The following is the conversation history:" +
// transcript). Setting `context` only appends, which cannot remove a section
// the built-in template demands — so it must be `prompt`.
//
// Because a `prompt` override also suppresses the built-in <prior-summary>
// embedding (`to = Ve.prompt ?? Aa({previousSummary, context})`), we fetch the
// last completed compaction ourselves so a second compaction in the same
// session still rolls forward. Messages come from `client.session.messages`
// (the SDK namespaces under `session`; there is no `client.message`).
//
// Fail-open: any error logs one line and leaves `prompt` untouched, so the
// stock OpenCode prompt runs.

interface CompactingInput {
  sessionID?: string
}

interface CompactingOutput {
  context?: string[]
  prompt?: string
}

const MERGE_RULES = `When a <prior-summary> is present, merge it with the conversation below:
- carry forward objective, constraints, decisions, workflow state and next moves even if the conversation no longer mentions them; drop only what is finished and no longer needed
- the conversation is newer than the prior summary — where they conflict the conversation wins: state the corrected fact and drop the old claim
- move finished work to Done, clear resolved blockers, refresh Objective / Workflow / Next Move`

const TEMPLATE = `Output exactly this Markdown, in this section order, with no preamble and no commentary:

## Objective
- [one line: what the user is finishing; name the docs/UX_FLOWS.md journey if one was named]

## Workflow
- Card: [<kanban card / dev-note id> — <status>]
- Branch: [<branch> · merged|unmerged>]
- Journey: [<UX_FLOWS journey, if named>]
- Gates: [lint · typecheck · tests <n/n> · bench <script|not run>]

## Details
- [constraints, decisions and why, exact paths / commands / error strings]

## State
- Done: [finished and verified work]
- Active: [current work, partial changes, investigation in flight]
- Blocked: [failing commands, unknowns, waiting on someone]

## Next Move
1. [immediate concrete action]
2. [next action if known, else omit]

Rules:
- Tight: at most 4 one-line bullets per section, no prose paragraphs.
- Skip a section (or a Workflow line) that was never mentioned; never write "(none)".
- Preserve exact file paths, symbols, commands, counts, error strings and URLs when known.
- No file-inventory section: paths belong in Details or Next Move.
- Never mention that context was compacted or that this is a summary.`

/** Text of the newest completed compaction message, or "" when there is none. */
function latestPriorSummary(messages: any[]): string {
  let bestTime = -1
  let bestText = ""
  for (const m of messages) {
    const info = m?.info ?? m
    if (info?.mode !== "compaction" && info?.summary !== true) continue
    const parts = Array.isArray(m?.parts) ? m.parts : []
    const text = parts
      .filter((p: any) => p?.type === "text" && typeof p.text === "string")
      .map((p: any) => p.text)
      .join("\n")
      .trim()
    if (!text) continue
    const created = info?.time?.created ?? 0
    if (created >= bestTime) {
      bestTime = created
      bestText = text
    }
  }
  return bestText
}

export default (async (input: { client?: any }) => {
  const client = input?.client

  return {
    "experimental.session.compacting": async (hookInput: CompactingInput, output: CompactingOutput) => {
      try {
        const sessionID = hookInput?.sessionID
        if (!sessionID || !output || typeof output !== "object") return

        const messagesFn = client?.session?.messages
        if (typeof messagesFn !== "function") {
          console.error("[compaction-tight] client.session.messages unavailable — using stock prompt")
          return
        }

        // `limit` caps at 9007199254740991 server-side; a single large page
        // avoids `before` pagination (not exposed on the SDK query type) and
        // keeps compaction messages — which sit near the front — in range.
        const res = await messagesFn.call(client.session, {
          path: { id: sessionID },
          query: { limit: 10000 },
        })
        const messages: any[] = Array.isArray(res?.data)
          ? res.data
          : Array.isArray(res)
            ? res
            : []

        const priorText = latestPriorSummary(messages)
        const prior = priorText
          ? `<prior-summary>\n${priorText}\n</prior-summary>\n\n${MERGE_RULES}`
          : ""

        output.prompt = [
          "Summarize the conversation appended below into an anchored handoff that another coding agent can act on without reading the transcript.",
          prior,
          TEMPLATE,
        ]
          .filter(Boolean)
          .join("\n\n")

        console.error(
          `[compaction-tight] prompt override applied (messages=${messages.length}, prior-summary=${priorText ? "yes" : "no"})`,
        )
      } catch (err) {
        // Fail open: leave `prompt` unset so the stock prompt runs.
        console.error("[compaction-tight] override failed, using stock prompt:", err)
      }
    },
  }
})
