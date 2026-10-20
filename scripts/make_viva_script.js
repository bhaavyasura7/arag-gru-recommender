// Builds docs/VIVA_SCRIPT.docx — the spoken script: presentation, live demo, concept cards, viva answers.
// Usage: npm install docx@9.5.1 (any scratch dir) && NODE_PATH=<that dir>/node_modules node scripts/make_viva_script.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, TableOfContents, LevelFormat,
  Footer, PageNumber, Header,
} = require("docx");

const ROOT = "/Users/likhith/Projects/DeepLearningProject";
const FIG = path.join(ROOT, "docs", "figures");
const W = 9026; // content width (A4, 1-inch margins), DXA
const FONT = "Calibri";
const MONO = "Consolas";

// ---------- inline markup: **bold**, `code`, _italic_ ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|_[^_]+_)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: 19, color: "1F4E79", ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const body = [];
const H1 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(t)] }));
const H2 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] }));
const H3 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] }));
const P = (t) => body.push(new Paragraph({ spacing: { after: 120 }, children: runs(t) }));
const B = (items, level = 0) => items.forEach((t) => body.push(new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 60 }, children: runs(t) })));
let numRef = 0;
const N = (items) => { const ref = `num${numRef++}`; numberingConfigs.push(numCfg(ref)); items.forEach((t) => body.push(new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children: runs(t) }))); };
const CODE = (lines) => lines.forEach((l, i) => body.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" },
  spacing: { after: i === lines.length - 1 ? 160 : 0 },
  indent: { left: 200, right: 200 },
  children: [new TextRun({ text: l || " ", font: MONO, size: 18 })],
})));
const NOTE = (title, text, fill = "EAF2FB", edge = "2E75B6") => body.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill, color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: edge, space: 8 } },
  spacing: { before: 120, after: 160 }, indent: { left: 200, right: 200 },
  children: [new TextRun({ text: title + "  ", bold: true, color: edge }), ...runs(text)],
}));
const IMG = (file, caption, widthIn = 6.2) => {
  const data = fs.readFileSync(path.join(FIG, file));
  const dims = [data.readUInt32BE(16), data.readUInt32BE(20)]; // PNG IHDR width, height
  const w = widthIn * 96, h = w * dims[1] / dims[0];
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({ type: "png", data, transformation: { width: w, height: h }, altText: { title: caption, description: caption, name: file } })] }));
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(caption, { italics: true, size: 19, color: "555555" }) }));
};
const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const TABLE = (header, rows, widths) => {
  const total = widths.reduce((a, b) => a + b, 0);
  const row = (cells, isHead) => new TableRow({ tableHeader: isHead, children: cells.map((c, i) => new TableCell(cellOpts(c, isHead, widths[i]))) });
  body.push(new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [row(header, true), ...rows.map((r) => row(r, false))] }));
  body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
};
function cellOpts(txt, isHead, width) {
  return {
    width: { size: width, type: WidthType.DXA },
    borders: { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder },
    shading: isHead ? { type: ShadingType.CLEAR, fill: "1F4E79", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(txt), isHead ? { bold: true, color: "FFFFFF", size: 19 } : { size: 19 }) })],
  };
}
let qn = 0;
const QA = (q, a) => {
  qn++;
  body.push(new Paragraph({ spacing: { before: 160, after: 60 }, keepNext: true, children: [new TextRun({ text: `Q${qn}. `, bold: true, color: "C00000" }), ...runs(q, { bold: true })] }));
  (Array.isArray(a) ? a : [a]).forEach((t) => body.push(new Paragraph({ spacing: { after: 80 }, indent: { left: 360 }, children: runs(t) })));
};
const numberingConfigs = [];
function numCfg(ref) {
  return { reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] };
}

// =====================================================================================
// SCRIPT-SPECIFIC HELPERS
// =====================================================================================
// SAY: a line to speak, SHOW: what to put on screen, DO: an action
const SAY = (t) => body.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: "F0F7EE", color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "548235", space: 8 } },
  spacing: { before: 60, after: 100 }, indent: { left: 200, right: 200 },
  children: [new TextRun({ text: "SAY  ", bold: true, color: "548235" }), ...runs(t)],
}));
const SHOW = (t) => body.push(new Paragraph({
  spacing: { before: 60, after: 60 }, indent: { left: 200 },
  children: [new TextRun({ text: "SHOW  ", bold: true, color: "2E75B6" }), ...runs(t, { color: "1F4E79" })],
}));
const DO = (t) => body.push(new Paragraph({
  spacing: { before: 60, after: 60 }, indent: { left: 200 },
  children: [new TextRun({ text: "DO  ", bold: true, color: "C55A11" }), ...runs(t, { color: "843C0C" })],
}));
const TIME = (t) => body.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "⏱ " + t, italics: true, color: "888888", size: 19 })] }));
const CONCEPT = (name, simple, analogy, viva) => {
  body.push(new Paragraph({ spacing: { before: 200, after: 60 }, keepNext: true, children: [new TextRun({ text: name, bold: true, size: 24, color: "1F4E79" })] }));
  body.push(new Paragraph({ spacing: { after: 60 }, keepNext: true, children: [new TextRun({ text: "In simple words: ", bold: true }), ...runs(simple)] }));
  if (analogy) body.push(new Paragraph({ spacing: { after: 60 }, keepNext: true, children: [new TextRun({ text: "Analogy: ", bold: true, color: "7F6000" }), ...runs(analogy, { italics: true })] }));
  SAY(viva);
};

// =====================================================================================
// TITLE PAGE
// =====================================================================================
body.push(new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "ARAG + GRU Recommender", bold: true, size: 56, color: "1F4E79" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "The Explanation Script", size: 36, color: "2E75B6" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Word-for-word lines for the presentation, the live demo and the viva questions", italics: true, size: 24 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600 }, children: [new TextRun({ text: "12-minute talk  ·  5-minute demo  ·  Every concept in 30 seconds  ·  33 tough questions  ·  Number cheat sheet", size: 21 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1600 }, children: [new TextRun({ text: "Companion to docs/PROJECT_GUIDE.docx (full explanations) and docs/paper/main.pdf", size: 20, color: "666666" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "October 2026", size: 20, color: "666666" })] }));

body.push(new Paragraph({ pageBreakBefore: true, children: [new TextRun({ text: "Contents", bold: true, size: 32, color: "1F4E79" })] }));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(new Paragraph({ children: runs("_If the table of contents is empty, right-click it in Word and choose “Update Field”._", { color: "888888", size: 18 }) }));

// =====================================================================================
H1("1. How to Use This Script");
P("This is the **speaking** version of the project. The Project Guide explains everything in depth; this document tells you **what to say, in what order, and what to show** while saying it.");
TABLE(["Marker", "Meaning"], [
  ["**SAY**  (green box)", "Words to speak. Learn the idea, not every word — say it in your own voice."],
  ["**SHOW**  (blue)", "What should be on screen: a figure, a page of the demo, a table."],
  ["**DO**  (orange)", "An action: click, type, scroll."],
  ["⏱", "Rough time for that part."],
], [2600, 6426]);
H2("1.1 The three things the examiner must leave with");
N([
  "**We rebuilt ARAG properly** and its claims hold: it beats the LLM methods the paper compared it with.",
  "**We tested what the paper didn’t**: properly trained sequence models (GRU, SASRec) are more accurate than ARAG alone; blending a GRU into ARAG significantly improves ARAG (+0.098) but doesn’t beat the GRU alone.",
  "**ARAG’s real value** is explanations, natural-language search and much lower popularity bias — and every number we show is audited, reproducible and costs $0 to re-run.",
]);
H2("1.2 Practice plan");
TABLE(["When", "What"], [
  ["Day 1", "Read Section 2 out loud twice with a timer. Aim for 12 minutes."],
  ["Day 2", "Rehearse the demo (Section 3) with the app running. Do it twice."],
  ["Day 3", "Concept cards (Section 4): cover the answer, explain each one in 30 seconds."],
  ["Day 4", "Tough questions (Section 5): ask a friend to fire them at you in random order."],
  ["Morning of the viva", "Number cheat sheet (Section 7) and the phrases to avoid (Section 6)."],
], [2200, 6826]);

// =====================================================================================
H1("2. The Presentation Script (≈ 12 minutes)");
P("Ten parts. Each one has the words to say and what to show. The figures are in `docs/figures/` and in the paper.");

H2("2.1 Opening");
TIME("30 seconds");
SAY("Good morning. Our project is about **recommending products with LLM agents**. We took a 2025 research paper from Walmart called **ARAG**, which uses four GPT agents to recommend products, rebuilt it from scratch on real Amazon data, and asked a simple question: **does adding a small trained neural network make the agents better?** On the way we also tested something the paper never tested — how ARAG compares with properly trained recommendation models.");

H2("2.2 The problem");
TIME("1 minute");
SAY("Every online store has to answer one question millions of times a day: **what should this customer see next?** That is a recommender system. The classic way is to train a model on what millions of people bought, in what order — ‘people who bought this next bought that’. It is accurate and cheap, but it can’t explain itself and it tends to push popular items.");
SAY("The new idea is to let a **large language model** do it: it reads a customer’s reviews like a human shop assistant would, understands what they like, and explains each recommendation in words. ARAG is the most advanced version of that idea. So the natural question is: **which is better, and can we combine them?**");

H2("2.3 What ARAG is — the four agents");
TIME("2 minutes");
SHOW("`docs/figures/fig01_pipeline.png` (the pipeline diagram).");
SAY("ARAG splits the job into **four agents**, each a GPT-4o-mini call with its own instructions. Think of a shop with four staff members:");
B([
  "**UUA — User Understanding Agent.** Reads the customer’s history and writes a short profile: ‘likes racing games on PlayStation, recently buying accessories, dislikes horror’.",
  "**NLI — Natural Language Inference agent.** Looks at each of the 20 candidate products and scores from 0 to 10 how well it matches that profile.",
  "**CSA — Context Summary Agent.** Takes the products that scored 5 or more and writes a short summary of why they fit.",
  "**IRA — Item Ranker Agent.** Reads both summaries and produces the final ranking of all 20 products, with a written reason for the top picks.",
]);
SAY("That is **7 GPT calls per customer**: UUA and the four NLI calls run in parallel, then CSA, then IRA. The ‘retrieval-augmented’ part is that the agents are given the customer’s real review history — title, category, star rating and date of each product — instead of answering from memory.");

H2("2.4 Our idea — blend in a GRU");
TIME("1 minute");
SAY("A **GRU** is a small recurrent neural network. We trained it on **625,000 reviews** to predict the next product from the order of past products. It costs nothing per request once trained. Our idea is a simple blend: **final score = α × ARAG’s score + (1 − α) × GRU’s score**. α is a number between 0 and 1 that we pick on separate validation customers — 0 means ‘GRU only’, 1 means ‘ARAG only’.");
SAY("We had two hypotheses. **H1:** the blend beats ARAG alone. **H2:** the blend beats the GRU alone.");

H2("2.5 Data and how we test fairly");
TIME("1½ minutes");
SAY("We use **Amazon Reviews 2023, Video Games category**: about **95,000 customers, 137,000 products and 815,000 reviews**. A review stands in for a purchase, which is standard practice because Amazon doesn’t publish purchase logs.");
SAY("The test works like this. For each customer we **hide their last review**. Then we build a list of **20 products: the hidden one plus 19 decoys** they never reviewed. The decoys are picked in proportion to popularity, so they look about as popular as the real one — a system can’t win just by choosing the popular item. Every system ranks the exact same 20, and we check how high it put the hidden one.");
SAY("The main score is **NDCG@5**: 1.0 if the hidden product is ranked first, 0.63 if second, 0.5 if third, down to 0 if it isn’t in the top 5. Random guessing scores about 0.13.");
SAY("LLM calls cost money, so the LLM systems are tested on **100 customers**; the trained models are free to run, so we test them on those same 100 **and** on all **94,762**. Every comparison has a **95% confidence interval** from 10,000 bootstrap resamples — if the interval of a difference doesn’t include zero, the difference is real.");
SHOW("Research page → “How we test” card.");

H2("2.6 What we compared");
TIME("1 minute");
TABLE(["Group", "Systems", "One-line description"], [
  ["Sanity checks", "Random, Popularity", "If a system can’t beat these, something is wrong"],
  ["Simple similarity", "Content similarity, ItemKNN", "‘Products like the ones you reviewed’ by text, or by co-occurrence"],
  ["Trained models", "GRU, text-aware GRU, SASRec", "Neural networks trained on purchase order; text-GRU also reads product descriptions; SASRec is a Transformer"],
  ["LLM only", "LLM-Recency, LLM-Vanilla-RAG, ARAG", "The paper’s two simpler LLM baselines, and ARAG itself"],
  ["Ours", "ARAG + GRU, ARAG + text-GRU", "The blends"],
], [1800, 2900, 4326]);
SAY("We added SASRec — a well-known Transformer recommender — as a check: if two different trained models agree, the numbers are trustworthy.");

H2("2.7 The results — seven questions");
TIME("3 minutes");
SHOW("Research page (`/#research`). Scroll down through the question cards as you speak.");
P("**Question 1 — Does adding the GRU help ARAG?**");
SAY("**Yes.** ARAG alone scores 0.424; ARAG plus GRU scores 0.523. The difference is **+0.098 with a 95% interval of +0.013 to +0.182** — it doesn’t include zero, so H1 is supported. The two make different kinds of mistakes, so combining them helps.");
P("**Question 2 — Does the blend beat the GRU alone?**");
SAY("**No — they are about the same.** +0.004, interval −0.038 to +0.045. So H2 is not supported. The chosen α was 0.2, meaning the ranking is mostly the GRU’s.");
P("**Question 3 — Which system is most accurate?**");
SAY("On the same 100 customers, **SASRec is top at 0.536**, our **ARAG + text-GRU blend is right behind at 0.532**, then ARAG + GRU and the GRU. ARAG alone is at 0.424 — it **beats the paper’s LLM baselines** (Recency 0.285, Vanilla-RAG 0.263) and the simple methods, but trails the trained models. ARAG versus SASRec is −0.112, a significant gap.");
P("**Question 4 — Does reading product descriptions help?**");
SAY("On all 94,762 customers, the **text-aware GRU is the best single model at 0.530**, versus 0.505 for the plain GRU and 0.508 for SASRec. With that many customers the intervals are about ±0.002, so these gaps are real.");
P("**Question 5 — So where does ARAG win?**");
SAY("**Popularity bias.** Trained models learn that popular products sell, so their top picks are much more popular than average — a popularity lift of about **0.77**. ARAG’s is **0.13**: it reads what the customer actually wrote and recommends beyond the bestsellers. It also covers more of the catalogue, and it explains every pick.");
P("**Question 6 — Do short histories hurt?**");
SAY("**Barely.** Most customers have only 4–6 reviews, so we checked. The GRU scores 0.497 for customers with 4 items and 0.519 for 11 or more — almost flat.");
P("**Question 7 — How was α chosen?**");
SAY("We tried α from 0 to 1 on **validation** customers only and picked the best, 0.2. The test curve has the same shape, so the choice generalises. We never tuned anything on test.");

H2("2.8 Mistakes we found and fixed");
TIME("1 minute");
SAY("Two honest stories. **First**, our original candidate lists drew decoys from the whole catalogue, most of which nobody had reviewed — popularity alone could spot the real item. We rebuilt the lists so decoys come only from products people actually reviewed, picked in proportion to popularity; now popularity scores 0.109, slightly below random, which proves the shortcut is gone.");
SAY("**Second**, our first GRU scored only 0.28 and we blamed short histories. When SASRec got stuck in the same way, we traced it to a **weight-initialisation bug**: PyTorch’s default embedding values were too large for our tied output layer, so training stalled. Setting a small initial scale fixed it — the GRU jumped to 0.52. We re-ran the entire evaluation from our saved LLM answers, so the fix cost **$0**, and we documented the old and new numbers side by side.");
SHOW("Project Guide §12 or REPORT.md Appendix G (before/after table).");

H2("2.9 Limitations and future work");
TIME("45 seconds");
B([
  "LLM systems were tested on **100 customers** because of cost, so small differences between them aren’t reliable — that’s why we always show intervals.",
  "One category (Video Games) and one LLM (gpt-4o-mini). A stronger LLM might close part of the gap.",
  "The paper compared ARAG only with LLM baselines; our trained-model comparison is an **addition**, not a contradiction.",
  "Future work: more categories, more test customers, and using review text inside the trained model.",
]);

H2("2.10 Closing");
TIME("30 seconds");
SAY("To sum up: **ARAG works as the paper says** and beats the LLM methods it was compared with. Blending in a trained GRU **significantly improves it**. But properly trained sequence models are **more accurate on their own and essentially free**, so ARAG’s real value is **explanations, natural-language search and lower popularity bias**. Every number we showed comes from a real run, is checked by an automatic audit, and can be reproduced for zero dollars. Thank you — we’re happy to show the live demo.");

// =====================================================================================
H1("3. The Live Demo Script (≈ 5 minutes)");
H2("3.1 Before the viva (10 minutes before)");
CODE([
  "cd ~/Projects/DeepLearningProject",
  "PORT=8010 make app            # live mode (needs OPENAI_API_KEY)",
  "# or, fully offline using only saved answers:",
  "REPLAY_ONLY=1 OPENAI_API_KEY=replay-only-dummy PORT=8010 make app",
  "",
  "# open  http://localhost:8010   and   http://localhost:8010/#research",
]);
B([
  "Open **two tabs**: the store and the Research page.",
  "Click through the customers and searches you plan to show **once beforehand**, so their answers are cached and load instantly (and work even offline).",
  "Have `docs/PROJECT_GUIDE.docx` and `docs/paper/main.pdf` open as backup.",
]);
H2("3.2 Step by step");
P("**Step 1 — Sign in as a customer.**");
DO("Click the account button (top right) and pick a customer with a clear taste, e.g. one whose main platform is PlayStation.");
SAY("These are **100 real Amazon customers** from the test set — the same 100 the LLM systems were evaluated on. Nothing here is fake data.");
P("**Step 2 — The history row.**");
SHOW("The row of past games at the top of “For you”.");
SAY("This is what this customer **actually reviewed**, in order. This is all the system knows about them.");
P("**Step 3 — The recommendations.**");
SHOW("The loading animation (UUA → NLI → CSA → IRA), then the cards.");
SAY("While it loads you can see the **four agents working** in order. Each card is ranked by ARAG blended with our text-aware GRU, and the sentence under each card is **ARAG’s reason**, written for this customer.");
DO("Click one card to open the product page.");
SAY("Here is the full product and the **evidence the agents used** — the parts of the customer’s history that made them recommend it.");
P("**Step 4 — Feedback.**");
DO("Dismiss one product, then refresh the feed.");
SAY("Dismissed products never come back. The GRU recomputes this customer’s state from the updated history **instantly — no retraining**.");
P("**Step 5 — Natural-language search.**");
DO("Type a query, e.g. “racing game for PS4 under 30 dollars” or “horror games but not zombies”.");
SAY("The LLM turns the sentence into keywords, a price limit and exclusions; FAISS finds matching products; then ARAG and the GRU re-rank them **for this customer**. A trained model alone can’t do this — this is where the LLM earns its cost.");
P("**Step 6 — Switch customer.**");
DO("Switch to a very different customer (e.g. an accessories buyer).");
SAY("Different person, visibly different page — the recommendations are genuinely personal.");
P("**Step 7 — Research page.**");
SHOW("`/#research`");
SAY("And this page shows the **same audited numbers as our report**, as seven questions with answers — I can go into any of them.");
H2("3.3 If something goes wrong");
TABLE(["Problem", "What to do / say"], [
  ["A page loads slowly", "“Each new page makes 7 GPT calls, about a third of a cent; cached pages are instant.” Switch to a customer you pre-loaded."],
  ["An error in offline mode", "“In replay mode only requests we’ve run before are available — that’s what guarantees zero cost.” Use a pre-loaded customer."],
  ["No internet / no API key", "Use REPLAY_ONLY mode with pre-loaded customers, or show the Research page and the figures."],
  ["The app won’t start at all", "Show `docs/paper/main.pdf` figures and the Research page screenshots in the Project Guide."],
], [2800, 6226]);

// =====================================================================================
H1("4. Every Concept in 30 Seconds");
P("Each card: the plain explanation, an everyday analogy, and the line to say.");
H2("4.1 The basics");
CONCEPT("Recommender system",
  "A program that ranks products for a person, based on what they and others did before.",
  "A shop assistant who remembers what you bought and suggests what to try next.",
  "A recommender ranks items for a user. Ours predicts the next product a customer will review, from their earlier reviews.");
CONCEPT("Next-item prediction",
  "Given someone’s history in order, guess the very next thing they’ll pick.",
  "Guessing the next song in someone’s playlist.",
  "We hide each customer’s last review and ask every system to rank it among 20 candidates.");
CONCEPT("LLM (large language model)",
  "A huge neural network trained on text that can read and write like a person. We use GPT-4o-mini.",
  "A very well-read assistant who has never seen your shop’s sales data.",
  "The LLM understands what customers write in reviews, but it has no knowledge of what customers in our data actually bought next.");
CONCEPT("RAG (retrieval-augmented generation)",
  "Before asking the LLM, look up the most relevant information and paste it into the prompt.",
  "Giving the assistant the customer’s file before they answer, instead of asking from memory.",
  "RAG fetches the relevant past reviews and product details so the LLM reasons over real evidence.");
CONCEPT("Agent",
  "An LLM call with one specific job and its own instructions; several agents pass results to each other.",
  "Four staff members, each doing one part of the job and handing notes to the next.",
  "ARAG uses four agents — understand the user, judge each item, summarise, rank — instead of one big prompt.");
H2("4.2 The models");
CONCEPT("ARAG",
  "Agentic RAG: retrieval + four cooperating LLM agents (UUA, NLI, CSA, IRA) that rank the candidates and explain why.",
  null,
  "ARAG replaces one big LLM prompt with four specialised agents; on our data it beats simpler LLM approaches by about 0.14–0.16 NDCG@5.");
CONCEPT("GRU (gated recurrent unit)",
  "A small recurrent neural network that reads a sequence of products one at a time and keeps a memory of what it has seen, then predicts the next product.",
  "Reading a story word by word and guessing the next word.",
  "Our GRU learns from 625,000 reviews which product tends to come next. It’s free at prediction time and doesn’t explain itself.");
CONCEPT("Text-aware GRU",
  "The same GRU, but each product is also represented by its title and description, not only its ID.",
  "Knowing not just ‘item #4521’ but ‘a racing game for PS4’.",
  "Adding description text makes it the best single model on all customers, 0.530 vs 0.505.");
CONCEPT("SASRec",
  "A standard Transformer-based recommender — it uses self-attention over the history instead of a recurrent memory.",
  null,
  "We added SASRec as a second, well-known trained model to check that our GRU numbers aren’t a fluke; it scores almost the same.");
CONCEPT("Late fusion (the blend)",
  "Each system scores the 20 products separately; we combine the scores: α × ARAG + (1 − α) × GRU.",
  "Asking two experts and averaging their opinions, trusting one more than the other.",
  "We blend after both systems have scored, with α chosen on validation data; α = 0.2 means the GRU gets most of the weight.");
CONCEPT("Embeddings / MiniLM",
  "Turning text into a list of numbers so that similar texts have similar numbers. MiniLM is a small model that does this.",
  "Giving every product a location on a map, where similar products are near each other.",
  "MiniLM turns each product’s title and description into 384 numbers. It powers the text-aware GRU, the content-similarity baseline, the Vanilla-RAG baseline and the demo search.");
CONCEPT("FAISS",
  "A library for finding the nearest vectors quickly.",
  null,
  "FAISS finds the products closest to a search query among 137,000 in milliseconds.");
H2("4.3 Evaluation");
CONCEPT("NDCG@5",
  "Our main score. Full points if the hidden product is ranked first, fewer the lower it is, zero if it’s outside the top 5. Averaged over customers.",
  "A race where 1st place gets 1.00, 2nd 0.63, 3rd 0.50, 4th 0.43, 5th 0.39, and everyone else 0.",
  "NDCG@5 rewards putting the right product near the very top, which is what a customer sees.");
CONCEPT("Hit@5 / Hit@10",
  "The share of customers whose hidden product is anywhere in the top 5 (or 10).",
  null,
  "Hit@k ignores the exact position — it just asks ‘was it on the first screen?’.");
CONCEPT("MRR",
  "Mean reciprocal rank: 1 divided by the position of the hidden product (1st = 1, 2nd = 0.5, 4th = 0.25), averaged.",
  null,
  "MRR looks at the full ranking, so it also rewards getting the item from 15th to 8th.");
CONCEPT("Bootstrap confidence interval",
  "Resample the customers 10,000 times with replacement and see how much the result moves. The middle 95% is the interval.",
  "Re-running the exam with a slightly different class each time and seeing how stable the average is.",
  "We use paired bootstrap intervals: if the interval of a difference excludes zero, the difference is statistically real.");
CONCEPT("Validation vs test",
  "Validation customers are used to choose settings (α, θ). Test customers are only used once, at the end, to report results.",
  "Practising on past papers, then sitting the real exam once.",
  "Nothing was tuned on test, so our test numbers are honest.");
CONCEPT("θ (theta)",
  "The NLI score threshold: candidates scoring at least θ (out of 10) are passed to the summary agent. Chosen on validation: θ = 5.",
  null,
  "θ = 5 was the best of 4–7 on validation.");
CONCEPT("Popularity bias",
  "How much more popular the recommended products are than the candidates on average. 0 = no bias.",
  "A shop assistant who only ever suggests the bestseller shelf.",
  "Trained models push popular products (about 0.77); ARAG barely does (0.13). That’s ARAG’s clearest advantage.");
CONCEPT("Diversity and coverage",
  "Diversity: how different the top-5 products are from each other. Coverage: how much of the catalogue ever gets recommended.",
  null,
  "ARAG covers slightly more of the catalogue (0.275 vs 0.252 for the GRU); diversity is similar for all systems.");

// =====================================================================================
H1("5. Tough Questions — Short Spoken Answers");
P("The Project Guide §14 has 65 questions with long answers. These are the ones most likely to be asked, with answers short enough to say in 20–40 seconds.");
H2("5.1 About the idea and the paper");
QA("Why did you choose this paper?", "It’s a recent, well-known agentic-RAG recommender from Walmart, it’s fully described so we could rebuild it, and it raises an obvious open question — how does it compare with trained models — that we could test on a student budget.");
QA("So is the paper wrong?", "No. Its claims hold on our data: ARAG beats the LLM baselines it was compared with by a clear, significant margin, and removing its agents hurts. What it didn’t include is a comparison with trained sequence models. We added that, and those are more accurate here. That’s an addition, not a contradiction.");
QA("Is ARAG the best system?", "Not on accuracy. ARAG alone is 0.424; SASRec is 0.536 and our blend 0.532. ARAG is best at low popularity bias, and it’s the only system that explains its picks and handles natural-language search.");
QA("Then why would anyone use ARAG?", "For explanations, conversational search and recommending beyond bestsellers. A sensible real system uses the trained model for ranking and the LLM for the reasons and the search — which is roughly what our demo does.");
QA("What is new in your work?", "Three things: a faithful re-implementation with audited numbers; the comparison with properly trained GRU and SASRec models that the paper didn’t run; and the fusion experiment showing the blend helps ARAG but not the GRU.");
H2("5.2 About the method");
QA("Why a GRU and not something bigger?", "It’s the classic sequential recommender, small and fast to train, which makes it a fair ‘cheap’ counterpart to the expensive LLM. We also added SASRec, a Transformer, and it scores almost the same, so the conclusion isn’t specific to GRUs.");
QA("Why blend scores instead of giving the GRU’s output to the LLM?", "Late fusion is simple, transparent and needs no extra LLM calls. Every value of α is evaluated from cached answers at zero cost, and α = 0 or 1 exactly reproduces each system on its own, so the comparison is clean.");
QA("How did you choose α?", "On 100 separate validation customers. We tried α from 0 to 1 in steps of 0.1 and picked the best, 0.2. The test set was never used for tuning.");
QA("Why only 100 test customers for the LLM?", "Cost and time: each customer needs about 7 GPT calls per system. We spent about $2 in total. To be fair, we report confidence intervals and never claim a winner from a small gap — and the trained models are also tested on all 94,762 customers.");
QA("Why 20 candidates and not the whole catalogue?", "An LLM can’t read 137,000 products in one prompt. The ARAG paper uses the same candidate-list setup, and giving every system the identical 20 makes the comparison fair.");
QA("How are the 19 negatives chosen?", "From the 25,527 products that appear in training, sampled in proportion to how popular they are, excluding anything the customer ever reviewed. So decoys are about as popular as real targets. Proof that it works: the popularity baseline scores 0.109, slightly below random at 0.125.");
QA("Why reviews instead of purchases?", "Amazon doesn’t publish purchase logs. Reviews are the standard public substitute in recommender research, and the original paper also uses Amazon review data (an older 2016 version).");
QA("Why gpt-4o-mini and not a bigger model?", "Budget and speed. The paper used gpt-3.5-turbo; gpt-4o-mini is newer, cheaper and generally stronger. A larger LLM might narrow the gap to the trained models, which we list as future work.");
H2("5.3 About the results");
QA("Is +0.098 for H1 significant?", "Yes. The 95% paired bootstrap interval is +0.013 to +0.182, which excludes zero.");
QA("Why is H2 not supported?", "The blend gets +0.004 over the GRU with an interval from −0.038 to +0.045 — it includes zero, so we can’t say there’s a real difference. With α = 0.2 the blend is mostly the GRU anyway.");
QA("Why does the blend help ARAG so much?", "They make different mistakes. The GRU knows what people usually buy next across the whole dataset; ARAG knows what this customer says they like. Combining two different views fixes many of each one’s errors.");
QA("SASRec 0.536 vs your blend 0.532 — who wins?", "Neither, statistically. The gap is 0.004 on 100 customers; anything under about 0.03 isn’t reliable at that size. We describe them as tied at the top.");
QA("Why is the text-aware GRU best on all customers but not on the 100?", "On the 100 it scores 0.501 vs 0.520 for the plain GRU, but 100 customers is a small, noisy sample. On all 94,762 customers the text-GRU is clearly ahead, 0.530 vs 0.505 with intervals of ±0.002, so we trust the large sample.");
QA("Did the agents actually matter?", "Yes. Removing the NLI and CSA agents drops ARAG by 0.062, which is significant; removing NLI alone drops it by 0.049. And ARAG beats the single-prompt LLM baselines by 0.14–0.16.");
QA("What does popularity lift 0.13 vs 0.77 mean in practice?", "The trained models’ top-5 picks are noticeably more popular than the average candidate — they lean on bestsellers. ARAG’s picks are almost as popular as a random candidate, so it surfaces less-known products that match the customer’s stated taste.");
QA("Do short histories hurt?", "Very little. The GRU scores 0.497 for customers with 4 reviews and 0.519 for 11 or more. This is also how we know the old ‘short histories’ explanation for the weak first GRU was wrong.");
H2("5.4 About the mistakes");
QA("Tell us about the GRU bug.", "PyTorch initialises embeddings with standard deviation 1. Our GRU uses the same embeddings for input and output, so the starting scores were huge, the softmax saturated and training stalled near random. Initialising with a small standard deviation of 0.02 — the usual choice for Transformer-style models — fixed it. The GRU went from 0.28 to 0.52.");
QA("How did you find it?", "We added SASRec, and its first training run got stuck at a loss around 32 with very low validation scores — the same symptom. That pointed to initialisation, and fixing it in both models solved it.");
QA("Doesn’t finding a bug make your results unreliable?", "It’s the opposite: we found it because we tested with a second model, we fixed it, re-ran everything from saved LLM answers for $0, and documented both old and new numbers. Every final number is checked by `make verify`.");
QA("Did the bug change your conclusions?", "Yes, and we say so openly. Before the fix the GRU looked weak and H1 wasn’t supported. After the fix H1 is supported, H2 isn’t, and trained models are more accurate than ARAG.");
QA("What was the candidate-list problem?", "The first lists drew decoys from the whole catalogue, and 81% of those had no interactions — so ‘pick the popular one’ found the target too easily. We rebuilt the lists from reviewed products only and re-ran everything.");
H2("5.5 About engineering and reproducibility");
QA("How do we know your numbers are right?", "Every number is produced by the pipeline and checked by `make verify`, which recomputes the metrics from the saved predictions. There are 59 automated tests, and every LLM answer is cached, so anyone can re-run the evaluation for $0.");
QA("How much did it cost?", "About **$2** of OpenAI usage in total for all evaluations. Everything after that — the fixes, SASRec, the extra analyses — was re-run from cached answers at $0. A live demo page costs about a third of a cent.");
QA("How long does ARAG take per customer?", "Seven sequential and parallel GPT calls — a few seconds per customer. The GRU takes milliseconds. That speed and cost difference is part of our conclusion.");
QA("Could this work in a real store?", "The architecture would: a trained model ranks quickly and cheaply, and the LLM agents add explanations and handle search queries. You’d run the agents only on the top few results to control cost.");
QA("What would you do with more time?", "Test more categories and many more customers, try a stronger LLM, and feed review text directly into the trained model.");
QA("What did you learn?", "That a simple, well-trained baseline is essential before claiming an LLM system wins, that you should always check results with a second model, and that being able to re-run everything cheaply makes fixing mistakes painless.");
QA("What was the hardest part?", "Making the comparison fair: identical candidate lists, tuning only on validation, paired confidence intervals — and then tracking down the initialisation bug.");

// =====================================================================================
H1("6. Phrases to Use and to Avoid");
TABLE(["Avoid saying", "Say instead", "Why"], [
  ["“The paper is wrong / proven inferior.”", "“The paper’s claims hold against the LLM baselines it tested; it didn’t test trained sequence models, and those are more accurate on our data.”", "Accurate and fair; examiners dislike overclaiming."],
  ["“ARAG is the best.”", "“ARAG is the best LLM-only method; trained models rank more accurately; ARAG’s strengths are explanations, search and low popularity bias.”", "Matches the numbers."],
  ["“Our blend beats SASRec.”", "“Our blend and SASRec are tied at the top.”", "0.004 gap on 100 customers isn’t significant."],
  ["“The GRU was bad because histories are short.”", "“The first GRU had an initialisation bug; history length barely matters.”", "The old explanation was disproved."],
  ["“We proved…”", "“Our results show… on this dataset, with a 95% confidence interval of…”", "One dataset, one LLM, 100 customers."],
  ["“It’s statistically significant” (for every gap)", "Check whether the interval excludes zero first.", "H2 and the SASRec–blend gap are not significant."],
], [2600, 4300, 2126]);

// =====================================================================================
H1("7. Number Cheat Sheet");
H2("7.1 Data");
TABLE(["Item", "Value"], [
  ["Dataset", "Amazon Reviews 2023 — Video Games (5-core)"],
  ["Customers / products / reviews", "94,762 / 137,269 / 814,586"],
  ["GRU training reviews", "625,062"],
  ["Candidates per customer", "20 (1 hidden + 19 popularity-matched decoys)"],
  ["Test customers", "100 for LLM systems; 94,762 for trained models"],
  ["Validation customers (tuning)", "100"],
  ["LLM", "gpt-4o-mini"],
  ["Chosen settings", "θ = 5, α = 0.2 (ARAG + GRU), α = 0.1 (ARAG + text-GRU)"],
  ["Total LLM cost", "≈ $2; all later re-runs $0"],
], [4000, 5026]);
H2("7.2 NDCG@5 on the same 100 customers");
TABLE(["System", "NDCG@5"], [
  ["SASRec", "**0.536**"],
  ["ARAG + text-GRU", "0.532"],
  ["ARAG + GRU", "0.523"],
  ["GRU (3-run average)", "0.520"],
  ["Text-aware GRU (3-run average)", "0.501"],
  ["Content similarity", "0.439"],
  ["**ARAG**", "**0.424**"],
  ["ItemKNN", "0.399"],
  ["LLM-Recency", "0.285"],
  ["LLM-Vanilla-RAG", "0.263"],
  ["Random", "0.125"],
  ["Popularity", "0.109"],
], [5000, 4026]);
H2("7.3 NDCG@5 on all 94,762 customers");
TABLE(["System", "NDCG@5"], [
  ["Text-aware GRU", "**0.530** ± 0.002"],
  ["SASRec", "0.508"],
  ["GRU", "0.505"],
  ["Content similarity", "0.411"],
  ["ItemKNN", "0.406"],
], [5000, 4026]);
H2("7.4 The key comparisons");
TABLE(["Comparison", "Difference [95% CI]", "Real?"], [
  ["H1: ARAG + GRU vs ARAG", "+0.098 [+0.013, +0.182]", "**Yes**"],
  ["H2: ARAG + GRU vs GRU", "+0.004 [−0.038, +0.045]", "No"],
  ["ARAG vs SASRec", "−0.112 [−0.213, −0.009]", "**Yes**"],
  ["ARAG vs GRU", "−0.095 [−0.196, +0.006]", "No (borderline)"],
  ["ARAG vs LLM-Recency / Vanilla-RAG", "+0.139 / +0.162", "**Yes**"],
  ["ARAG without NLI & CSA vs ARAG", "−0.062", "**Yes**"],
  ["Text-GRU vs GRU (all customers)", "+0.026 [+0.024, +0.027]", "**Yes**"],
], [3800, 3400, 1826]);
H2("7.5 Beyond accuracy (100 customers)");
TABLE(["", "ARAG", "Trained models"], [
  ["Popularity lift (lower = less bias)", "**0.13**", "≈ 0.77"],
  ["Catalogue coverage", "**0.275**", "≈ 0.25"],
  ["History length (GRU, 4 items vs 11+)", "—", "0.497 vs 0.519 (almost flat)"],
], [4200, 2400, 2426]);
H2("7.6 The bug, before and after");
TABLE(["", "Before fix", "After fix"], [
  ["GRU NDCG@5 (all customers)", "0.279", "0.505"],
  ["GRU NDCG@5 (100 customers, run 0)", "0.238", "0.519"],
  ["H1 (blend vs ARAG)", "not supported", "**supported** (+0.098)"],
], [4000, 2500, 2526]);
NOTE("Last line before you walk in.", "“ARAG works as published; trained models are more accurate and nearly free; blending helps ARAG but not the GRU; ARAG’s value is explanations, search and lower popularity bias — and every number is audited and reproducible.”", "F0F7EE", "548235");

const doc = new Document({
  creator: "ARAG + GRU project team",
  title: "ARAG + GRU Recommender — Explanation Script",
  description: "Presentation script, demo script, concept cards and viva answers",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 34, bold: true, color: "1F4E79", font: FONT }, paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 27, bold: true, color: "2E75B6", font: FONT }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, color: "404040", font: FONT }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 300 } } } },
      ] },
      ...numberingConfigs,
    ],
  },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "ARAG + GRU — Explanation Script", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18, color: "888888" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "docs", "VIVA_SCRIPT.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes;", qn, "questions");
});
