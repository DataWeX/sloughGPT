/**
 * AML (Automatic Markup Language) parser and serializer for TypeScript.
 *
 * Grammar (simplified):
 *   document    := header? block*
 *   header      := '@aml' VERSION NEWLINE
 *   block       := TAG NAME? ('{' body '}') | TAG NAME '=' value
 *   body        := (assignment | list_item | nested_block)*
 *   assignment  := IDENT '=' value
 *   list_item   := '-' value
 *   nested_block:= TAG NAME? '{' body '}'
 */

export type AmlScalar = string | number | boolean | null;

export interface AmlBlock {
  tag: string;
  name: string | null;
  body: AmlValue | null;
  metadata: Record<string, AmlValue>;
  line: number;
}

export interface AmlDocument {
  version: string;
  blocks: AmlBlock[];
  errors: string[];
  byTag(tag: string): AmlBlock[];
  byName(name: string): AmlBlock | undefined;
}

// ── regex patterns ───────────────────────────────────────────────────

const RE_COMMENT = /^\s*#/;
const RE_HEADER = /^@aml\s+(\d+\.\d+)/;
const RE_BLANK = /^\s*$/;
const RE_TAG_WITH_BRACE = /^\s*(@\w+)(?:\s+([^{]+))?\s*\{(.*)\}\s*$/;
const RE_TAG_OPEN = /^\s*(@\w+)(?:\s+([^{]+))?\s*\{\s*$/;
const RE_INLINE_EQ = /^\s*(@\w+)\s+(\S+)\s*=\s*(.+)$/;
const RE_ASSIGN = /^\s+(\w+)\s*=\s*(.+)$/;
const RE_LIST_ITEM = /^\s*-\s+(.+)$/;

// ── value parser ──────────────────────────────────────────────────────

function splitCSV(s: string): string[] {
  const parts: string[] = [];
  let inQuote: string | null = null;
  let current: string[] = [];

  for (const ch of s) {
    if ((ch === '"' || ch === "'") && (inQuote === null || inQuote === ch)) {
      if (inQuote === ch) inQuote = null;
      else inQuote = ch;
      current.push(ch);
    } else if (ch === "," && inQuote === null) {
      parts.push(current.join("").trim());
      current = [];
    } else {
      current.push(ch);
    }
  }
  if (current.length > 0) {
    parts.push(current.join("").trim());
  }
  return parts;
}

export type AmlValue = AmlScalar | AmlScalar[];

export function parseValue(raw: string): AmlValue {
  const stripped = raw.trim();

  // inline list: [a, b, c]
  if (stripped.startsWith("[") && stripped.endsWith("]")) {
    const inner = stripped.slice(1, -1).trim();
    if (!inner) return [] as AmlScalar[];
    return splitCSV(inner).flatMap((item) => {
      const v = parseValue(item.trim());
      return Array.isArray(v) ? v : [v];
    });
  }

  // quoted string
  if (
    (stripped.startsWith('"') && stripped.endsWith('"')) ||
    (stripped.startsWith("'") && stripped.endsWith("'"))
  ) {
    return stripped.slice(1, -1);
  }

  // null
  if (stripped.toLowerCase() === "null" || stripped.toLowerCase() === "none" || stripped === "") {
    return null;
  }

  // bool
  if (stripped.toLowerCase() === "true" || stripped.toLowerCase() === "yes" || stripped.toLowerCase() === "on") {
    return true;
  }
  if (stripped.toLowerCase() === "false" || stripped.toLowerCase() === "no" || stripped.toLowerCase() === "off") {
    return false;
  }

  // int
  if (/^-?\d+$/.test(stripped)) {
    return parseInt(stripped, 10);
  }

  // float
  if (/^-?\d+\.\d+$/.test(stripped)) {
    return parseFloat(stripped);
  }

  // unquoted string
  return stripped;
}

// ── parser ────────────────────────────────────────────────────────────

function parseBlockBody(
  lines: string[],
  start: number,
  tag: string,
  name: string | null,
  doc: AmlDocument
): { block: AmlBlock; nextIndex: number } {
  const n = lines.length;
  let i = start;
  let depth = 1;
        const metadata: Record<string, AmlValue> = {};
  const listItems: AmlScalar[] = [];

  while (i < n && depth > 0) {
    const bl = lines[i].trimEnd();

    // skip blank / comment
    if (RE_BLANK.test(bl) || RE_COMMENT.test(bl)) {
      i++;
      continue;
    }

    // list item: - value
    const lm = bl.match(RE_LIST_ITEM);
    if (lm) {
      listItems.push(parseValue(lm[1].trim()) as AmlScalar);
      i++;
      continue;
    }

    // assignment: key = value
    const am = bl.match(RE_ASSIGN);
    if (am) {
      metadata[am[1]] = parseValue(am[2]) as AmlScalar;
      i++;
      continue;
    }

    // inline assignment: @tag name = value (skip inside body)
    const iem = bl.match(RE_INLINE_EQ);
    if (iem) {
      i++;
      continue;
    }

    // multi-line nested block: @tag name {
    const nom = bl.match(RE_TAG_OPEN);
    if (nom) {
      let nestedTag = nom[1].replace(/^@/, "");
      let nestedName = nom[2] ? nom[2].trim() : null;
      const { block: nested, nextIndex } = parseBlockBody(lines, i + 1, nestedTag, nestedName, doc);
      doc.blocks.push(nested);
      i = nextIndex;
      continue;
    }

    // single-line nested block: @tag name { key = val }
    const snm = bl.match(RE_TAG_WITH_BRACE);
    if (snm && bl.includes("{") && bl.includes("}")) {
      const nestedTag = snm[1].replace(/^@/, "");
      const nestedName = snm[2] ? snm[2].trim() : null;
      const inner = snm[3].trim();
      const nestedMeta: Record<string, AmlValue> = {};
      if (inner) {
        for (const part of inner.split(",")) {
          const trimmed = part.trim();
          const assignmentMatch = ("    " + trimmed).match(RE_ASSIGN);
          if (assignmentMatch) {
            nestedMeta[assignmentMatch[1]] = parseValue(assignmentMatch[2]) as AmlScalar;
          }
        }
      }
      doc.blocks.push({
        tag: nestedTag,
        name: nestedName,
        body: null,
        metadata: nestedMeta,
        line: i + 1,
      });
      i++;
      continue;
    }

    // track brace depth for non-nested lines
    for (const ch of bl) {
      if (ch === "{") depth++;
      else if (ch === "}") depth--;
    }
    if (depth <= 0) {
      i++;
      break;
    }

    i++;
  }

  // build body
  let body: AmlValue | null = null;
  if (listItems.length > 0) {
    body = listItems;
  }

  return {
    block: { tag, name, body, metadata, line: start },
    nextIndex: i,
  };
}

export function parse(source: string): AmlDocument {
  const doc: AmlDocument = {
    version: "1.0",
    blocks: [],
    errors: [],
    byTag(tag: string) {
      return this.blocks.filter((b) => b.tag === tag);
    },
    byName(name: string) {
      return this.blocks.find((b) => b.name === name);
    },
  };

  const lines = source.split("\n");
  let i = 0;
  const n = lines.length;

  while (i < n) {
    const line = lines[i];

    // skip blank lines and comments
    if (RE_BLANK.test(line) || RE_COMMENT.test(line)) {
      i++;
      continue;
    }

    // header: @aml 1.0
    const hm = line.match(RE_HEADER);
    if (hm) {
      doc.version = hm[1];
      i++;
      continue;
    }

    // inline assignment: @tag name = value
    const iem = line.match(RE_INLINE_EQ);
    if (iem) {
      const tag = iem[1].replace(/^@/, "");
      const name = iem[2];
      const val = parseValue(iem[3]);
      doc.blocks.push({ tag, name, body: val, metadata: {}, line: i + 1 });
      i++;
      continue;
    }

    // block: @tag name { ... }  (single or multi-line)
    let m = line.match(RE_TAG_OPEN);
    if (!m) m = line.match(RE_TAG_WITH_BRACE);
    if (m) {
      const tag = m[1].replace(/^@/, "");
      let name = m[2] ? m[2].trim() : null;

      // single-line block: @tag name { key = val }
      if (line.match(RE_TAG_WITH_BRACE) && line.includes("{") && line.includes("}")) {
        const braceStart = line.indexOf("{");
        const braceEnd = line.lastIndexOf("}");
        const inner = line.slice(braceStart + 1, braceEnd).trim();
  const metadata: Record<string, AmlValue> = {};
        if (inner) {
          for (const part of inner.split(",")) {
            const trimmed = part.trim();
            const assignmentMatch = ("    " + trimmed).match(RE_ASSIGN);
            if (assignmentMatch) {
              metadata[assignmentMatch[1]] = parseValue(assignmentMatch[2]) as AmlScalar;
            }
          }
        }
        doc.blocks.push({ tag, name, body: null, metadata, line: i + 1 });
        i++;
        continue;
      }

      // multi-line block: parse body until matching }
      const { block, nextIndex } = parseBlockBody(lines, i + 1, tag, name, doc);
      doc.blocks.push(block);
      i = nextIndex;
      continue;
    }

    // skip unrecognized lines
    i++;
  }

  return doc;
}

// ── serializer ────────────────────────────────────────────────────────

function formatValue(val: AmlValue, inList = false): string {
  if (val === null) return "null";
  if (typeof val === "boolean") return val ? "true" : "false";
  if (typeof val === "number") return String(val);
  if (Array.isArray(val)) {
    const inner = val.map((item) => formatValue(item)).join(", ");
    return `[${inner}]`;
  }
  if (typeof val === "string") {
    const needsQuote =
      /[={}\n\t]/.test(val) ||
      val !== val.trim() ||
      (!inList && val.includes(" "));
    if (needsQuote) {
      const escaped = val.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
      return `"${escaped}"`;
    }
    return val;
  }
  return String(val);
}

function serializeBlock(block: AmlBlock, indent: string): string {
  const lines: string[] = [];
  const tag = `@${block.tag}`;
  const header = block.name ? `${tag} ${block.name}` : tag;

  // inline body (no metadata, simple value)
  if (block.body !== null && Object.keys(block.metadata).length === 0) {
    if (block.body === null || typeof block.body !== "object") {
      lines.push(`${header} = ${formatValue(block.body)}`);
      return lines.join("\n");
    }
  }

  // block body
  lines.push(`${header} {`);

  if (block.body !== null) {
    if (Array.isArray(block.body)) {
      for (const item of block.body) {
        lines.push(`${indent}- ${formatValue(item, true)}`);
      }
    } else if (typeof block.body === "string") {
      for (const part of block.body.split("\n")) {
        lines.push(`${indent}${part}`);
      }
    }
  }

  for (const [key, val] of Object.entries(block.metadata)) {
    lines.push(`${indent}${key} = ${formatValue(val)}`);
  }

  lines.push("}");
  return lines.join("\n");
}

export function serialize(doc: AmlDocument, indent = "    "): string {
  const lines: string[] = [];

  // header
  lines.push(`@aml ${doc.version}`);
  lines.push("");

  for (const block of doc.blocks) {
    lines.push(serializeBlock(block, indent));
    lines.push("");
  }

  return lines.join("\n") + "\n";
}

// ── dict → AML conversion ────────────────────────────────────────────

export function dictToAml(data: Record<string, unknown>, version = "1.0"): string {
  const doc = createDocument(version);

  for (const [key, val] of Object.entries(data)) {
    let tag = key;
    let name: string | null = null;
    if (key.includes(":")) {
      [tag, name] = key.split(":", 2);
    }

    if (typeof val === "object" && val !== null && !Array.isArray(val)) {
      const obj = val as Record<string, unknown>;
      const body = obj["body"] ?? null;
      const { body: _, ...meta } = obj;
      doc.blocks.push({
        tag,
        name,
        body: body as AmlValue | null,
        metadata: meta as Record<string, AmlValue>,
        line: 0,
      });
    } else if (Array.isArray(val)) {
      doc.blocks.push({
        tag,
        name,
        body: val as AmlScalar[],
        metadata: {},
        line: 0,
      });
    } else {
      doc.blocks.push({
        tag,
        name,
        body: val as AmlScalar,
        metadata: {},
        line: 0,
      });
    }
  }

  return serialize(doc);
}

// ── JSON → AML conversion ────────────────────────────────────────────

export function jsonToAml(json: string, version = "1.0"): string {
  const data = JSON.parse(json) as Record<string, unknown>;
  return dictToAml(data, version);
}

// ── factory ───────────────────────────────────────────────────────────

export function createDocument(version = "1.0"): AmlDocument {
  return {
    version,
    blocks: [],
    errors: [],
    byTag(tag: string) {
      return this.blocks.filter((b) => b.tag === tag);
    },
    byName(name: string) {
      return this.blocks.find((b) => b.name === name);
    },
  };
}
