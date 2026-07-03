import { describe, it, expect } from "vitest";
import {
  parse,
  serialize,
  parseValue,
  dictToAml,
  jsonToAml,
  createDocument,
} from "./aml";

describe("parseValue", () => {
  it("parses quoted strings", () => {
    expect(parseValue('"hello world"')).toBe("hello world");
    expect(parseValue("'hello'")).toBe("hello");
  });

  it("parses numbers", () => {
    expect(parseValue("42")).toBe(42);
    expect(parseValue("3.14")).toBe(3.14);
    expect(parseValue("-7")).toBe(-7);
  });

  it("parses booleans", () => {
    expect(parseValue("true")).toBe(true);
    expect(parseValue("false")).toBe(false);
    expect(parseValue("yes")).toBe(true);
    expect(parseValue("no")).toBe(false);
  });

  it("parses null", () => {
    expect(parseValue("null")).toBe(null);
    expect(parseValue("none")).toBe(null);
    expect(parseValue("")).toBe(null);
  });

  it("parses inline lists", () => {
    expect(parseValue("[a, b, c]")).toEqual(["a", "b", "c"]);
    expect(parseValue("[1, 2, 3]")).toEqual([1, 2, 3]);
    expect(parseValue("[]")).toEqual([]);
  });

  it("parses unquoted strings", () => {
    expect(parseValue("hello")).toBe("hello");
    expect(parseValue("hello world")).toBe("hello world");
  });
});

describe("parse", () => {
  it("parses header", () => {
    const doc = parse("@aml 1.0\n");
    expect(doc.version).toBe("1.0");
    expect(doc.blocks).toHaveLength(0);
  });

  it("parses inline assignment", () => {
    const doc = parse("@tag name = value\n");
    expect(doc.blocks).toHaveLength(1);
    expect(doc.blocks[0].tag).toBe("tag");
    expect(doc.blocks[0].name).toBe("name");
    expect(doc.blocks[0].body).toBe("value");
  });

  it("parses single-line block", () => {
    const doc = parse("@a { x = 1, y = 2 }\n");
    expect(doc.blocks).toHaveLength(1);
    expect(doc.blocks[0].tag).toBe("a");
    expect(doc.blocks[0].metadata).toEqual({ x: 1, y: 2 });
  });

  it("parses multi-line block", () => {
    const doc = parse("@chapter intro {\n  title = Introduction\n  content = Hello\n}\n");
    expect(doc.blocks).toHaveLength(1);
    expect(doc.blocks[0].tag).toBe("chapter");
    expect(doc.blocks[0].metadata.title).toBe("Introduction");
    expect(doc.blocks[0].metadata.content).toBe("Hello");
  });

  it("parses nested blocks", () => {
    const doc = parse(`@aml 1.0

@chapter biology {
    @section mitochondria {
        content = The powerhouse
    }

    @section photosynthesis {
        content = Plants make food
    }
}
`);
    expect(doc.blocks).toHaveLength(3);
    expect(doc.blocks[0].tag).toBe("section");
    expect(doc.blocks[0].name).toBe("mitochondria");
    expect(doc.blocks[1].tag).toBe("section");
    expect(doc.blocks[1].name).toBe("photosynthesis");
    expect(doc.blocks[2].tag).toBe("chapter");
    expect(doc.blocks[2].name).toBe("biology");
  });

  it("parses list items", () => {
    const doc = `@list items {
  - apple
  - banana
  - cherry
}
`;
    const parsed = parse(doc);
    expect(parsed.blocks).toHaveLength(1);
    expect(parsed.blocks[0].body).toEqual(["apple", "banana", "cherry"]);
  });

  it("skips comments", () => {
    const doc = `# this is a comment
@a { x = 1 }
# another comment
`;
    const parsed = parse(doc);
    expect(parsed.blocks).toHaveLength(1);
  });

  it("parses multiple single-line blocks", () => {
    const doc = parse("@a { x = 1 }\n@b { y = 2 }\n@a { z = 3 }\n");
    expect(doc.blocks).toHaveLength(3);
    expect(doc.blocks[0].metadata).toEqual({ x: 1 });
    expect(doc.blocks[1].metadata).toEqual({ y: 2 });
    expect(doc.blocks[2].metadata).toEqual({ z: 3 });
  });
});

describe("serialize", () => {
  it("serializes header", () => {
    const doc = createDocument("1.0");
    const output = serialize(doc);
    expect(output).toContain("@aml 1.0");
  });

  it("serializes inline assignment", () => {
    const doc = createDocument("1.0");
    doc.blocks.push({
      tag: "tag",
      name: "name",
      body: "value",
      metadata: {},
      line: 0,
    });
    const output = serialize(doc);
    expect(output).toContain("@tag name = value");
  });

  it("serializes block with metadata", () => {
    const doc = createDocument("1.0");
    doc.blocks.push({
      tag: "chapter",
      name: "intro",
      body: null,
      metadata: { title: "Introduction", content: "Hello" },
      line: 0,
    });
    const output = serialize(doc);
    expect(output).toContain("@chapter intro {");
    expect(output).toContain("title = Introduction");
    expect(output).toContain("content = Hello");
    expect(output).toContain("}");
  });

  it("serializes list body", () => {
    const doc = createDocument("1.0");
    doc.blocks.push({
      tag: "list",
      name: "items",
      body: ["apple", "banana", "cherry"],
      metadata: {},
      line: 0,
    });
    const output = serialize(doc);
    expect(output).toContain("- apple");
    expect(output).toContain("- banana");
    expect(output).toContain("- cherry");
  });

  it("round-trips parse → serialize → parse", () => {
    const src = `@aml 1.0

@knowledge mitochondria {
    content = "The mitochondria is the powerhouse."
    topic = "biology"
    tags = ["cell", "energy"]
}
`;
    const doc1 = parse(src);
    const serialized = serialize(doc1);
    const doc2 = parse(serialized);
    expect(doc2.blocks).toHaveLength(1);
    expect(doc2.blocks[0].tag).toBe("knowledge");
    expect(doc2.blocks[0].name).toBe("mitochondria");
    expect(doc2.blocks[0].metadata.content).toBe("The mitochondria is the powerhouse.");
    expect(doc2.blocks[0].metadata.topic).toBe("biology");
  });
});

describe("dictToAml", () => {
  it("converts dict to AML", () => {
    const aml = dictToAml({
      "knowledge:mitochondria": {
        content: "The mitochondria is the powerhouse.",
        topic: "biology",
      },
    });
    expect(aml).toContain("@aml 1.0");
    expect(aml).toContain("@knowledge mitochondria {");
    expect(aml).toContain('content = "The mitochondria is the powerhouse."');
    expect(aml).toContain("topic = biology");
  });
});

describe("jsonToAml", () => {
  it("converts JSON to AML", () => {
    const aml = jsonToAml(
      '{"knowledge:mind": {"content": "Mind is powerful"}}'
    );
    expect(aml).toContain("@knowledge mind {");
    expect(aml).toContain('content = "Mind is powerful"');
  });
});

describe("byTag / byName", () => {
  it("finds blocks by tag", () => {
    const doc = parse("@a x = 1\n@b y = 2\n@a z = 3\n");
    expect(doc.byTag("a")).toHaveLength(2);
    expect(doc.byTag("b")).toHaveLength(1);
    expect(doc.byTag("c")).toHaveLength(0);
  });

  it("finds block by name", () => {
    const doc = parse("@a foo { x = 1 }\n@b bar { y = 2 }\n");
    expect(doc.byName("foo")?.tag).toBe("a");
    expect(doc.byName("bar")?.tag).toBe("b");
    expect(doc.byName("baz")).toBeUndefined();
  });
});
