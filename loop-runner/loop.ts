#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import yaml from "js-yaml";
import { query } from "@anthropic-ai/claude-agent-sdk";

type ClaudeAgentOptions = Record<string, any>;
type Message = Record<string, any>;

const __filename = fileURLToPath(import.meta.url);
const RUNNER_DIR = path.dirname(__filename);
const ROOT = path.resolve(RUNNER_DIR, "..");
const CONFIG_PATH = path.join(RUNNER_DIR, "config", "loop.yaml");
const CFG = yaml.load(fs.readFileSync(CONFIG_PATH, "utf8")) as any;

const PROMPT_VALUES: Record<string, string> = {
  PROJECT_NAME: CFG.project.name,
  RESULTS_DIR: CFG.paths.results_dir,
  MANUSCRIPT_MAIN: CFG.paths.manuscript_main,
  MANUSCRIPT_SUP: CFG.paths.manuscript_sup,
  BIB_PATH: CFG.paths.references_bib,
  SPG_PATH: CFG.paths.spg,
  OUTLINE_PATH: CFG.paths.outline,
  DOCS_DIR: CFG.paths.docs_dir,
  OUTLINE_STRATEGY: CFG.planner.outline_strategy,
  IMAGES_DIR: CFG.paths.images_dir,
  REVIEW_DIR: "article/artifacts",
  AUTHOR_GUIDELINES: "",
  ARXIV_DIR: CFG.paths.arxiv_dir ?? "article/arxiv/papers",
  LITERATURE_INDEX: CFG.paths.literature_index ?? "article/arxiv/index.json",
};

const STEP_INDEX: Record<string, number> = {
  research: 1,
  codex: 2,
  claude: 3,
  gemini_audit: 4,
  gemini: 5,
};

type RunResult = { output: string; code: number };

function renderTemplate(filePath: string): string {
  let text = fs.readFileSync(filePath, "utf8");
  Object.entries(PROMPT_VALUES).forEach(([key, value]) => {
    text = text.replace(new RegExp(`{{${key}}}`, "g"), String(value));
  });
  return text;
}

function loadAuthorGuidelines(): string[] {
  const dir = CFG.paths.guidelines_dir;
  if (!dir) return [];
  const abs = path.join(ROOT, dir);
  if (!fs.existsSync(abs)) return [];
  const files = fs.readdirSync(abs).filter((f) => f.endsWith(".md"));
  const chunks: string[] = [];
  for (const file of files.sort()) {
    try {
      const txt = fs.readFileSync(path.join(abs, file), "utf8").trim();
      if (txt) chunks.push(txt);
    } catch {
      continue;
    }
  }
  return chunks;
}

function latestBlob(prefix: string): [string, string] | null {
  const dir = path.join(ROOT, "article", "artifacts");
  if (!fs.existsSync(dir)) return null;
  const candidates = fs
    .readdirSync(dir)
    .filter((f) => f.startsWith(prefix) && f.endsWith(".json"))
    .map((f) => path.join(dir, f));
  if (!candidates.length) return null;
  const latest = candidates.sort(
    (a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs,
  )[0];
  try {
    return [path.basename(latest), fs.readFileSync(latest, "utf8").trim()];
  } catch {
    return null;
  }
}

const latestReviewBlob = () => latestBlob("review_");
const latestCitationAuditBlob = () => latestBlob("citation_audit_");

function handleJsonLine(line: string): boolean {
  try {
    const obj = JSON.parse(line);
    const msg = obj?.message;
    const content = msg?.content ?? obj?.content;
    const parts: string[] = [];
    if (Array.isArray(content)) {
      for (const p of content) {
        if (p?.type === "text" && p.text) parts.push(p.text);
      }
    } else if (typeof content === "string" && content.trim()) {
      parts.push(content.trim());
    }
    if (parts.length) {
      console.log(parts.join("\n"));
      return true;
    }
  } catch {
    return false;
  }
  return false;
}

function runCommand(cmd: string[], opts: { cwd?: string } = {}): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd[0], cmd.slice(1), {
      cwd: opts.cwd ?? ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    let buffer = "";
    const handleChunk = (chunk: Buffer) => {
      const str = chunk.toString();
      output += str;
      buffer += str;
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        if (!handleJsonLine(line)) {
          console.log(line);
        }
      }
    };
    proc.stdout.on("data", handleChunk);
    proc.stderr.on("data", handleChunk);
    proc.on("close", (code) => {
      if (buffer.trim()) {
        if (!handleJsonLine(buffer)) console.log(buffer.trim());
        output += buffer;
      }
      resolve({ output, code: code ?? 0 });
    });
    proc.on("error", reject);
  });
}

function extractReviewFromStream(streamText: string): string | null {
  if (!streamText) return null;
  const assembled: string[] = [];
  for (const line of streamText.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const obj = JSON.parse(trimmed);
      if (obj?.type === "message" && obj?.role === "assistant") {
        const content = obj.content;
        if (typeof content === "string" && content) assembled.push(content);
        if (Array.isArray(content)) {
          for (const part of content) {
            if (part?.type === "text" && part.text) assembled.push(part.text);
          }
        }
      }
    } catch {
      continue;
    }
  }
  if (!assembled.length) return null;
  const payload = assembled.join("");
  if (payload.includes("```json")) {
    const start = payload.indexOf("```json") + "```json".length;
    const end = payload.indexOf("```", start);
    const body = payload.slice(start, end === -1 ? undefined : end).trim();
    return body;
  }
  return payload.trim() || null;
}

function parseJsonPayload(payload: string): any {
  const cleaned = (payload || "").trim();
  if (!cleaned) throw new Error("empty payload");
  try {
    return JSON.parse(cleaned);
  } catch {
    const lines = cleaned.split(/\r?\n/);
    return JSON.parse(lines[lines.length - 1]);
  }
}

async function runCodex(promptPath: string) {
  const prompt = renderTemplate(promptPath);
  await runCommand([
    "codex",
    "exec",
    "--cd",
    ROOT,
    "--json",
    "--",
    prompt,
  ]);
}

async function runClaudeDraft() {
  let prompt = renderTemplate(path.join(RUNNER_DIR, "prompts", "claude_draft.txt"));
  const review = latestReviewBlob();
  const audit = latestCitationAuditBlob();
  if (review) {
    prompt += `\n\n---\nLatest Gemini review (\`${review[0]}\`):\n\`\`\`json\n${review[1]}\n\`\`\`\nAddress each point (or log TODOs) before handing off.\n`;
  }
  if (audit) {
    prompt += `\n\n---\nLatest citation audit (\`${audit[0]}\`):\n\`\`\`json\n${audit[1]}\n\`\`\`\nResolve misplacements/missing citations where possible; otherwise leave explicit TODOs.\n`;
  }
  await runCommand([
    "claude",
    "--verbose",
    "--allow-dangerously-skip-permissions",
    "--dangerously-skip-permissions",
    "--permission-mode",
    "acceptEdits",
    "-p",
    "--output-format",
    "stream-json",
    prompt,
  ]);
}

async function runGemini(promptPath: string, extraContext = ""): Promise<string> {
  const base = renderTemplate(promptPath);
  const prompt = extraContext ? `${base}\n\n${extraContext}` : base;
  const { output } = await runCommand([
    "gemini",
    "--output-format",
    "stream-json",
    prompt,
  ], { cwd: path.join(ROOT, "article") });
  return output;
}

function buildResearchOptions(): ClaudeAgentOptions {
  let mcpServers: Record<string, any> = {};
  const mcpPath = path.join(ROOT, ".mcp.json");
  if (fs.existsSync(mcpPath)) {
    try {
      const raw = JSON.parse(fs.readFileSync(mcpPath, "utf8"));
      if (raw?.mcpServers && typeof raw.mcpServers === "object") {
        mcpServers = raw.mcpServers;
      }
    } catch {
      mcpServers = {};
    }
  }
  const debugDir = process.env.CLAUDE_DEBUG_DIR || path.join(ROOT, ".claude", "debug");
  try {
    fs.mkdirSync(debugDir, { recursive: true });
  } catch {
    // ignore
  }
  return {
    model: "sonnet-4-5",
    permissionMode: "acceptEdits",
    allowedTools: [
      "Read(article/backbone/**)",
      "Read(article/manuscript/**)",
      "Read(article/docs/**)",
      "Read(article/arxiv/**)",
      "Write(article/arxiv/index.json)",
      "Write(article/arxiv/papers/**)",
      "WebSearch",
      "WebFetch",
      "mcp:arxiv/*",
    ],
    disallowedTools: [
      "Edit(**/*)",
      "Bash(**/*)",
      "NotebookEdit(**/*)",
      "Write(**/*)",
    ],
    cwd: ROOT,
    mcpServers,
    settingSources: ["project"],
    env: {
      ...process.env,
      CLAUDE_DEBUG_DIR: debugDir,
    },
    pathToClaudeCodeExecutable: process.env.CLAUDE_PATH || "claude",
    stderr: (data: string) => {
      const trimmed = data.trim();
      if (trimmed) console.error(trimmed);
    },
  };
}

async function runClaudeResearch(): Promise<any> {
  const promptPath = path.join(RUNNER_DIR, "prompts", "claude_research.txt");
  const prompt = renderTemplate(promptPath);
  const opts = buildResearchOptions();
  const messages: Message[] = [];
  for await (const msg of query({ prompt, options: opts })) {
    messages.push(msg as Message);
    const content = (msg as any).content;
    if (Array.isArray(content)) {
      const texts = content
        .filter((p) => p?.type === "text" && p.text)
        .map((p) => p.text);
      if (texts.length) console.log(texts.join("\n"));
    } else if (typeof content === "string" && content.trim()) {
      console.log(content.trim());
    }
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    const content = (messages[i] as any).content;
    if (Array.isArray(content)) {
      for (const block of content) {
        if (block?.type === "text" && block.text) {
          try {
            const parsed = JSON.parse(block.text);
            return parsed;
          } catch {
            continue;
          }
        }
      }
    } else if (typeof content === "string" && content.trim()) {
      try {
        return JSON.parse(content.trim());
      } catch {
        continue;
      }
    }
  }
  return { session_id: "", fetched: [], skipped: [], todos: [] };
}

function applyAuthorGuidelines(extra: string[]) {
  const combined = extra.filter(Boolean).join("\n\n");
  PROMPT_VALUES.AUTHOR_GUIDELINES = combined || "(no explicit author guidelines provided)";
}

function readInlineArgs(): { startFrom: string; buildPdf: boolean; guidelines: string[]; guidelineFiles: string[] } {
  const args = process.argv.slice(2);
  const guidelines: string[] = [];
  const guidelineFiles: string[] = [];
  let startFrom = "research";
  let buildPdf = false;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--start-from" && args[i + 1]) {
      startFrom = args[++i];
    } else if (arg === "--build-pdf") {
      buildPdf = true;
    } else if (arg === "--guideline" && args[i + 1]) {
      guidelines.push(args[++i]);
    } else if (arg === "--guideline-file" && args[i + 1]) {
      guidelineFiles.push(args[++i]);
    }
  }
  return { startFrom, buildPdf, guidelines, guidelineFiles };
}

function readGuidelineFiles(files: string[]): string[] {
  const chunks: string[] = [];
  for (const file of files) {
    const abs = path.isAbsolute(file) ? file : path.join(ROOT, file);
    try {
      const txt = fs.readFileSync(abs, "utf8").trim();
      if (txt) chunks.push(txt);
    } catch {
      continue;
    }
  }
  return chunks;
}

async function main() {
  const { startFrom, buildPdf, guidelines, guidelineFiles } = readInlineArgs();
  const startIndex = STEP_INDEX[startFrom] ?? STEP_INDEX.research;

  const guidelineChunks = [
    ...guidelines.map((g) => g.trim()).filter(Boolean),
    ...readGuidelineFiles(guidelineFiles),
    ...loadAuthorGuidelines(),
  ];
  applyAuthorGuidelines(guidelineChunks);

  const manuscripts = [
    path.join(ROOT, PROMPT_VALUES.MANUSCRIPT_MAIN),
    path.join(ROOT, PROMPT_VALUES.MANUSCRIPT_SUP),
    path.join(ROOT, PROMPT_VALUES.BIB_PATH),
  ];
  const original: Record<string, string | null> = {};
  for (const m of manuscripts) {
    try {
      original[m] = fs.readFileSync(m, "utf8");
    } catch {
      original[m] = null;
    }
  }

  const timings: Record<string, number> = {
    research: 0,
    codex_graph: 0,
    codex_outline: 0,
    claude: 0,
    build_pdf: 0,
    gemini_audit: 0,
    gemini: 0,
  };
  const overallStart = Date.now();
  const artifactsDir = path.join(ROOT, "article", "artifacts");
  fs.mkdirSync(artifactsDir, { recursive: true });

  const gates = CFG.readability_gates ?? {};
  const clarityMin = Number(gates.clarity_min ?? 0);
  const storytellingMin = Number(gates.storytelling_min ?? 0);
  const readabilityMin = Number(gates.readability_min ?? 0);
  const maxConf = Number(gates.max_confusions ?? Number.MAX_SAFE_INTEGER);
  const maxStyle = Number(gates.max_style_findings ?? Number.MAX_SAFE_INTEGER);

  const iters = Number(CFG.loop.max_iters ?? 1);

  for (let i = 1; i <= iters; i++) {
    console.log(`\n==== LOOP ${i} ====\n`);
    const loopStart = i === 1 ? startIndex : STEP_INDEX.research;
    let auditPayload: string | null = null;

    if (loopStart <= STEP_INDEX.research) {
      const t0 = Date.now();
      console.log("[loop] Running Claude Research (SDK)");
      const researchJson = await runClaudeResearch();
      const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
      const p = path.join(artifactsDir, `research_loop${i}_${ts}.json`);
      fs.writeFileSync(p, JSON.stringify(researchJson, null, 2));
      timings.research += (Date.now() - t0) / 1000;
    } else {
      console.log("[loop] Skipping Claude research stage (start-from>=codex).");
    }

    if (loopStart <= STEP_INDEX.codex) {
      const t0 = Date.now();
      console.log("[loop] Running Codex Graph");
      await runCodex(path.join(RUNNER_DIR, "prompts", "codex_graph.txt"));
      timings.codex_graph += (Date.now() - t0) / 1000;
      const t1 = Date.now();
      console.log("[loop] Running Codex Outline");
      await runCodex(path.join(RUNNER_DIR, "prompts", "codex_linearize.txt"));
      timings.codex_outline += (Date.now() - t1) / 1000;
    } else {
      console.log("[loop] Skipping Codex stages (start-from>=claude).");
    }

    if (loopStart <= STEP_INDEX.claude) {
      const t0 = Date.now();
      console.log("[loop] Running Claude Draft");
      await runClaudeDraft();
      timings.claude += (Date.now() - t0) / 1000;
    } else {
      console.log("[loop] Skipping Claude stage (start-from>=gemini_audit).");
    }

    if (buildPdf) {
      const t0 = Date.now();
      console.log("[loop] Building PDF");
      await runCommand(["latexmk", "-pdf", "-quiet", PROMPT_VALUES.MANUSCRIPT_MAIN], { cwd: ROOT });
      timings.build_pdf += (Date.now() - t0) / 1000;
    }

    if (loopStart <= STEP_INDEX.gemini_audit) {
      const t0 = Date.now();
      console.log("[loop] Running Gemini Citation Audit");
      const auditRaw = await runGemini(path.join(RUNNER_DIR, "prompts", "gemini_citation_audit.txt"));
      const extracted = extractReviewFromStream(auditRaw);
      const audit = parseJsonPayload(extracted ?? auditRaw);
      auditPayload = JSON.stringify(audit, null, 2);
      const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
      fs.writeFileSync(path.join(artifactsDir, `citation_audit_loop${i}_${ts}.json`), auditPayload);
      timings.gemini_audit += (Date.now() - t0) / 1000;
    } else {
      console.log("[loop] Skipping Gemini citation audit (start-from=gemini).");
      const latest = latestCitationAuditBlob();
      if (latest) auditPayload = latest[1];
    }

    const auditContext = auditPayload
      ? `---\nCitation audit findings preceding this review:\n\`\`\`json\n${auditPayload}\n\`\`\`\nIncorporate these findings; ensure misplacements/missing citations are reflected in your feedback.\n`
      : "";

    const t0 = Date.now();
    console.log("[loop] Running Gemini Blind Review");
    const reviewRaw = await runGemini(
      path.join(RUNNER_DIR, "prompts", "gemini_blind_review.txt"),
      auditContext,
    );
    const extractedReview = extractReviewFromStream(reviewRaw);
    const review = parseJsonPayload(extractedReview ?? reviewRaw);
    const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
    fs.writeFileSync(path.join(artifactsDir, `review_loop${i}_${ts}.json`), JSON.stringify(review, null, 2));
    timings.gemini += (Date.now() - t0) / 1000;

    const confs = (review.confusions ?? []).length;
    const clarity = Number(review.clarity_score ?? 0);
    const storytelling = Number(review.storytelling_score ?? 0);
    const readability = Number(review.readability_score ?? 0);
    const styleCount = Array.isArray(review.style_findings) ? review.style_findings.length : 0;
    if (
      CFG.loop.stop_on_success &&
      clarity >= clarityMin &&
      storytelling >= storytellingMin &&
      readability >= readabilityMin &&
      confs <= maxConf &&
      styleCount <= maxStyle
    ) {
      console.log(
        `[loop] stop: clarity=${clarity} storytelling=${storytelling} readability=${readability} confusions=${confs} style_findings=${styleCount}`,
      );
      break;
    }
  }

  console.log("\n==== Manuscript Diff Summary ====");
  for (const m of manuscripts) {
    const rel = path.relative(ROOT, m);
    const before = original[m];
    let after: string | null = null;
    try {
      after = fs.readFileSync(m, "utf8");
    } catch {
      after = null;
    }
    if (before == null && after == null) {
      console.log(`- ${rel}: unavailable`);
      continue;
    }
    if (before == null) {
      const added = after ? after.split(/\r?\n/).length : 0;
      console.log(`- ${rel}: created (${added} lines)`);
      continue;
    }
    if (after == null) {
      console.log(`- ${rel}: deleted`);
      continue;
    }
    if (before === after) {
      console.log(`- ${rel}: no textual changes`);
      continue;
    }
    console.log(`- ${rel}: changed (diff not shown)`);
  }

  const totalElapsed = (Date.now() - overallStart) / 1000;
  console.log("\n==== Timing Summary ====");
  console.log(`Total elapsed: ${totalElapsed.toFixed(1)}s`);
  Object.entries(timings).forEach(([k, v]) => console.log(`- ${k}: ${v.toFixed(1)}s`));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
