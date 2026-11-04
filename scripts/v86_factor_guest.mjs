#!/usr/bin/env node
// scripts/v86_factor_guest.mjs — dumb serial driver for
// scripts/benchmark_v86_factor.py (card 20261103_121721, C1).
//
// Boots v86 headless in Node, injects the static workload over serial
// (base64, wrapped at 76 cols — the canonical tty input queue is 4095
// bytes), md5-verifies it, runs it, and reports HOST wall-clock seconds
// for the run: the guest clock runs on emulated TSC time and is not
// comparable to the host's. The guest's own elapsed time is emitted as
// a diagnostic only.
//
// Flow control: the tty echoes input, so an echoed line proves the
// emulator handed every byte of it to the guest — batches wait on their
// last line's echo; a failed md5 falls back to line-at-a-time ACKs.
//
// Contract: ONE JSON line to stdout ({guest_s, method, ...}); logs to
// stderr. Exit 0 on success, non-zero with an error JSON otherwise.
import { V86 } from "v86"
import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"

const argv = {}
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, "")
  argv[key] = process.argv[i + 1]
}
const { wasm, kernel, bios, "vga-bios": vgaBios, binary } = argv
const TIMEOUT_MS = (Number(argv.timeout) || 300) * 1000
const CMDLINE =
  "earlyprintk=serial,ttyS0,115200 tsc=reliable mitigations=off random.trust_cpu=on"
const WORKLOAD = "/tmp/v86bm"
const MARKER = "V86FACTOR_OK"
const PROMPT = "~% "

for (const [name, value] of Object.entries({ wasm, kernel, bios, vgaBios, binary })) {
  if (!value) {
    console.log(JSON.stringify({ error: `missing --${name}` }))
    process.exit(1)
  }
}

const buf = readFileSync(binary)
const localMd5 = createHash("md5").update(buf).digest("hex")
const b64 = buf.toString("base64")
const wrapped = b64.replace(/(.{76})/g, "$1\n").replace(/\n$/, "")
const payloadLines = wrapped.split("\n")

let raw = ""
let flat = "" // raw minus CR/LF — tty echo wraps at ~80 cols, content is contiguous
const t0 = Date.now()
const elapsed = () => ((Date.now() - t0) / 1000).toFixed(1)

function fail(phase, extra = {}) {
  console.error(`[driver] FAIL ${phase} @${elapsed()}s`)
  console.log(JSON.stringify({ error: phase, tail: raw.slice(-500), ...extra }))
  process.exit(1)
}

function waitFor(pred, ms, what) {
  return new Promise((resolve, reject) => {
    const started = Date.now()
    const poll = setInterval(() => {
      if (pred()) {
        clearInterval(poll)
        resolve()
      } else if (Date.now() - started > ms) {
        clearInterval(poll)
        reject(new Error(`timeout waiting for ${what} after ${ms}ms`))
      }
    }, 25)
  })
}

const emulator = new V86({
  wasm_path: wasm,
  memory_size: 256 * 1024 * 1024,
  vga_memory_size: 8 * 1024 * 1024,
  bios: { url: bios },
  vga_bios: { url: vgaBios },
  bzimage: { url: kernel },
  cmdline: CMDLINE,
  autostart: true,
})
emulator.add_listener("serial0-output-byte", (b) => {
  const ch = String.fromCharCode(b)
  raw += ch
  if (ch !== "\r" && ch !== "\n") flat += ch
})

async function sendLine(line, from, waitMs) {
  emulator.serial0_send(line + "\n")
  await waitFor(() => flat.slice(from).includes(line), waitMs, `echo of ${line.slice(0, 40)}`)
}

async function guestMd5() {
  const from = flat.length
  emulator.serial0_send(`md5sum ${WORKLOAD}\n`)
  await waitFor(() => /([0-9a-f]{32})/.test(flat.slice(from)), 15000, "md5sum output")
  return flat.slice(from).match(/([0-9a-f]{32})/)[1]
}

async function injectLineByLine() {
  let from = flat.length
  emulator.serial0_send(`rm -f ${WORKLOAD}\n`)
  await waitFor(() => flat.slice(from).includes(PROMPT), 5000, "prompt after rm")
  from = flat.length
  emulator.serial0_send("PS2=''\n")
  await waitFor(() => flat.slice(from).includes(PROMPT), 5000, "prompt after PS2 (retry)")
  from = flat.length
  emulator.serial0_send(`base64 -d > ${WORKLOAD} << 'V86EOF'\n`)
  await waitFor(() => flat.slice(from).includes(PROMPT) || flat.slice(from).includes("V86EOF"), 5000, "heredoc open")
  for (const line of payloadLines) {
    await sendLine(line, flat.length, 5000)
  }
  from = flat.length
  emulator.serial0_send("V86EOF\n")
  await waitFor(() => flat.slice(from).includes(PROMPT), 10000, "prompt after heredoc (retry)")
  from = flat.length
  emulator.serial0_send(`chmod +x ${WORKLOAD}\n`)
  await waitFor(() => flat.slice(from).includes(PROMPT), 5000, "prompt after chmod (retry)")
  return flat.length
}

async function inject() {
  // Fast path: open the heredoc, then batches of 100 lines, ACK on the
  // last line's echo.
  const BATCH = 100
  try {
    let from = flat.length
    emulator.serial0_send("PS2=''\n")
    await waitFor(() => flat.slice(from).includes(PROMPT), 5000, "prompt after PS2")
    from = flat.length
    emulator.serial0_send(`base64 -d > ${WORKLOAD} << 'V86EOF'\n`)
    await waitFor(() => flat.slice(from).includes("V86EOF"), 5000, "heredoc open")
    for (let i = 0; i < payloadLines.length; i += BATCH) {
      const batch = payloadLines.slice(i, i + BATCH)
      from = flat.length
      emulator.serial0_send(batch.join("\n") + "\n")
      const last = batch[batch.length - 1]
      await waitFor(() => flat.slice(from).includes(last), 10000, `batch ${i / BATCH + 1}`)
    }
    from = flat.length
    emulator.serial0_send("V86EOF\n")
    await waitFor(() => flat.slice(from).includes(PROMPT), 10000, "prompt after heredoc")
    from = flat.length
    emulator.serial0_send(`chmod +x ${WORKLOAD}\n`)
    await waitFor(() => flat.slice(from).includes(PROMPT), 5000, "prompt after chmod")
  } catch (e) {
    // The tty input queue is 4095 bytes — if the guest could not keep up,
    // fall back to line-at-a-time ACKs (in-flight: one line).
    console.error(`[driver] batch path failed (${e.message}), retrying line-by-line`)
    await injectLineByLine()
    const seen = await guestMd5()
    if (seen !== localMd5) fail("md5 after line-by-line retry", { localMd5, seen })
    return { method: "base64-line-ack" }
  }

  const seen = await guestMd5()
  if (seen === localMd5) return { method: "base64-batch100" }

  // A dropped byte corrupts the file — retry line-at-a-time.
  console.error(`[driver] md5 mismatch (${seen} != ${localMd5}), retrying line-by-line`)
  await injectLineByLine()
  const seen2 = await guestMd5()
  if (seen2 !== localMd5) fail("md5 after line-by-line retry", { localMd5, seen: seen2 })
  return { method: "base64-line-ack" }
}

async function main() {
  const bootFrom = Date.now()
  await waitFor(() => flat.includes(PROMPT), 90000, "shell prompt")
  const bootS = (Date.now() - bootFrom) / 1000
  console.error(`[driver] shell prompt @${elapsed()}s (boot ${bootS.toFixed(1)}s)`)

  const { method } = await inject()
  console.error(`[driver] injected via ${method} @${elapsed()}s`)

  const runFrom = flat.length
  const runT0 = performance.now()
  emulator.serial0_send(`${WORKLOAD}\n`)
  const runTimeout = Math.max(10000, TIMEOUT_MS - (Date.now() - t0))
  await waitFor(() => flat.slice(runFrom).includes(MARKER), runTimeout, "workload marker")
  const guestS = (performance.now() - runT0) / 1000

  const markerLine = flat.slice(runFrom).match(new RegExp(`${MARKER} [0-9.]+ [0-9.eE+-]+`))
  const guestClock = markerLine ? Number(markerLine[0].split(" ")[1]) : null

  console.log(
    JSON.stringify({
      guest_s: guestS,
      method,
      boot_s: Number(bootS.toFixed(2)),
      guest_clock_us: guestClock,
      md5_ok: true,
      bytes: buf.length,
    })
  )
  process.exit(0)
}

setTimeout(() => fail("overall-timeout"), TIMEOUT_MS)
main().catch((e) => fail(`driver: ${e.message}`))
