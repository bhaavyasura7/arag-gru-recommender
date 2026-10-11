// Builds docs/PROJECT_GUIDE.docx — the complete learning guide + viva Q&A.
// Usage: npm install docx@9.5.1 (any scratch dir) && node scripts/make_guide.js
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
  const dims = { fig02: [2174, 987], fig03: [1892, 894], fig04: [1864, 974], fig05: [1574, 894], fig07: [2174, 947], fig08: [1674, 894], fig10: [2574, 947], fig11: [1674, 894] }[file.slice(0, 5)];
  const w = widthIn * 96, h = w * dims[1] / dims[0];
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)), transformation: { width: w, height: h }, altText: { title: caption, description: caption, name: file } })] }));
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
// TITLE PAGE
// =====================================================================================
body.push(new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "ARAG + GRU Recommender", bold: true, size: 56, color: "1F4E79" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "The Complete Project Guide", size: 36, color: "2E75B6" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Everything the team needs to understand, run, demo and defend the project", italics: true, size: 24 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600 }, children: [new TextRun({ text: "Concepts from scratch  ·  Data & protocol  ·  Every model explained  ·  Code walkthrough", size: 21 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Running the pipeline  ·  Testing the frontend  ·  Results  ·  Professor Q&A bank", size: 21 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1600 }, children: [new TextRun({ text: "Companion to docs/paper/main.pdf (academic report) and docs/REPORT.md", size: 20, color: "666666" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "October 2026", size: 20, color: "666666" })] }));

body.push(new Paragraph({ pageBreakBefore: true, children: [new TextRun({ text: "Contents", bold: true, size: 32, color: "1F4E79" })] }));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(new Paragraph({ children: runs("_If the table of contents is empty, right-click it in Word and choose “Update Field”._", { color: "888888", size: 18 }) }));

// =====================================================================================
H1("1. How to Use This Guide");
P("This guide is written so that every team member can explain any part of the project without having written that part. Read it in order the first time; afterwards use it as a reference.");
TABLE(["If you have…", "Read"], [
  ["10 minutes", "Section 2 (the story) and Section 10 (results)"],
  ["1 hour", "Sections 2–6 and 10–11"],
  ["The night before the viva", "Section 14 (Q&A bank) twice, then Section 10"],
  ["To run the demo", "Sections 8 and 9"],
  ["To change code", "Section 7 (code walkthrough)"],
], [3000, 6026]);
NOTE("Golden rule.", "Every number in this project comes from a real run and can be re-derived with `make verify`. If a professor asks “how do you know this number is right?”, that is the answer — and it is the strongest point of the project.");

// =====================================================================================
H1("2. The Big Picture in Plain Words");
H2("2.1 The one-paragraph story");
P("We built a recommender system for video-game products from Amazon. Given what a person bought before, it must guess what they buy next. We compared two very different ways of doing this. The **first** is an **LLM agent system (ARAG)**: GPT reads the person’s history like a human would and ranks the options, using four cooperating “agents”. The **second** is a **trained neural network (a GRU)** that learns patterns from 625,000 past purchases. Our original question was: **does adding the trained GRU to the LLM system make it better?** The answer was **no** — the LLM system was clearly better on its own. We then discovered **why**: people in this dataset have very short histories (typically only 4 purchases), so a network that only sees product ID numbers has almost nothing to learn from. We fixed this by letting the GRU also read the product descriptions (through text embeddings). This **text-aware GRU** jumped from 0.279 to 0.461 on our main metric and **matched or beat the expensive LLM system — at zero API cost**.");
H2("2.2 The three research questions");
TABLE(["Question", "Short answer"], [
  ["RQ1: Does fusing a trained GRU into ARAG improve ranking (H1)? Does ARAG add over the GRU (H2)?", "H1 **no** (−0.013, not significant). H2 **yes** (+0.174, significant)."],
  ["RQ2: Do the original ARAG paper’s claims hold on our data?", "**Yes.** ARAG beats the paper’s simpler LLM baselines, and removing its NLI+CSA agents significantly hurts."],
  ["RQ3: Why did the GRU fail, and can we fix it without spending on the LLM?", "Short histories. Adding item text to the GRU lifts it **0.279 → 0.461** NDCG@5, at **$0**."],
], [4500, 4526]);
H2("2.3 The five things to remember");
N([
  "**Fair comparison:** every system ranks the *same* 20 candidate products per user (1 real next purchase + 19 decoys).",
  "**ARAG is a strong LLM system**: NDCG@5 0.424, significantly better than the paper’s simpler LLM baselines.",
  "**The plain GRU was weak** (0.279) because histories are short; adding it to ARAG did not help.",
  "**The text-aware GRU fixes it** (0.461 on all 94,762 users) and is at least as good as ARAG, for free.",
  "**We caught and corrected our own mistake**: an early version of the candidate lists was unfair and gave a false positive result. We documented it openly (Section 12).",
]);

// =====================================================================================
H1("3. Background Concepts from Scratch");
H2("3.1 Recommender systems");
P("A recommender system ranks items (products, films, songs) for a user. **Collaborative filtering** uses patterns across many users (“people who bought X also bought Y”). **Content-based** methods use item descriptions (“this is also a racing game for Wii U”). **Sequential recommendation** cares about _order_: what you bought most recently matters most.");
H2("3.2 The next-item prediction task");
P("We use **leave-last-out**: for each user, their **last** purchase is hidden as the test answer, the **second-to-last** is hidden for validation (tuning), and everything earlier is training history. The system sees the history and must rank candidates so the hidden item comes as high as possible.");
H2("3.3 Embeddings");
P("An **embedding** is a list of numbers (a vector) that represents something. Similar things get similar vectors. Two kinds appear in this project:");
B([
  "**ID embeddings** — learned by the GRU during training, one 64-number vector per product ID. They only become good if the product appears in many training sequences.",
  "**Text embeddings** — produced by **MiniLM** (`all-MiniLM-L6-v2`), a pre-trained sentence encoder that turns a product’s title/category/description into a 384-number vector. Products with similar descriptions get similar vectors even if no one ever bought them together.",
]);
P("Similarity between two normalised vectors is their **dot product** (cosine similarity): 1 = same direction, 0 = unrelated.");
H2("3.4 Recurrent networks and the GRU");
P("A **recurrent neural network (RNN)** reads a sequence one step at a time and keeps a **hidden state** — a running summary of what it has seen. A **GRU (Gated Recurrent Unit)** is an RNN with two gates that decide how much to update the summary at each step:");
CODE([
  "update gate   z_t = σ(W_z·x_t + U_z·h_{t−1})",
  "reset gate    r_t = σ(W_r·x_t + U_r·h_{t−1})",
  "candidate     h̃_t = tanh(W·x_t + U·(r_t ⊙ h_{t−1}))",
  "new state     h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t",
]);
P("Here `x_t` is the embedding of the item at step t, σ is the sigmoid function (outputs between 0 and 1) and ⊙ is element-wise multiplication. The gates let the GRU remember important older items and forget irrelevant ones. **GRU4Rec** (Hidasi et al., 2016) was the first well-known use of GRUs for recommendation; our model follows it.");
H2("3.5 Softmax and cross-entropy (how the GRU learns)");
P("After reading items 1…t, the GRU produces a user vector `u_t`. It scores **every** product j in the vocabulary by `u_t · e(j)`. **Softmax** turns these scores into probabilities that sum to 1. **Cross-entropy loss** is −log(probability of the item that actually came next). Training (Adam optimiser) adjusts all weights to make the true next item more probable, at every position of every user’s history. This is called a **full-softmax next-item objective**.");
H2("3.6 Weight tying");
P("The same embedding table is used for the **input** (reading history items) and the **output** (scoring candidates). This halves the parameters and means an item’s “meaning” is consistent in both roles.");
H2("3.7 Large language models, RAG and agents");
B([
  "**LLM** — a model like GPT that reads and writes text. We use OpenAI **gpt-4o-mini** at temperature 0 (as deterministic as possible).",
  "**Prompting** — giving the LLM instructions plus data (the user’s history and the candidates) and asking for an answer.",
  "**RAG (Retrieval-Augmented Generation)** — first retrieve relevant information, then give it to the LLM. “Vanilla RAG” here = pick the 10 history items most similar to the candidates.",
  "**Agents** — several LLM calls, each with a specialised job, whose outputs feed each other. ARAG uses four agents (Section 6.1).",
  "**Strict JSON schema** — we force the LLM to answer in a fixed machine-readable format, then validate it.",
]);
H2("3.8 FAISS");
P("**FAISS** (Facebook AI Similarity Search) is a library for finding the nearest vectors quickly. We use an exact inner-product index (`IndexFlatIP`) over the MiniLM item vectors. It powers the demo’s feed and search and the Vanilla-RAG baseline.");
H2("3.9 Evaluation metrics with worked numbers");
P("There is exactly one correct item per user. Let r be its position in a system’s ranking (1 = top).");
TABLE(["Metric", "Formula", "r=1", "r=2", "r=3", "r=5", "r=6"], [
  ["Hit@5", "1 if r ≤ 5 else 0", "1", "1", "1", "1", "0"],
  ["NDCG@5", "1/log₂(r+1) if r ≤ 5 else 0", "1.000", "0.631", "0.500", "0.387", "0"],
  ["MRR", "1/r", "1.000", "0.500", "0.333", "0.200", "0.167"],
], [1200, 2826, 1000, 1000, 1000, 1000, 1000]);
P("All metrics are averaged over users. **Hit@5** asks “is it in the top 5?”. **NDCG@5** also rewards putting it higher. **Random ranking** of 20 items gives Hit@5 = 5/20 = 0.25 on average — our sanity anchor.");
H2("3.10 Paired bootstrap (how we decide “significant”)");
P("Is ARAG really better than the GRU, or just lucky on these 100 users? For each user we compute the difference in NDCG@5 between the two systems. We then **resample the 100 users with replacement 10,000 times**, each time computing the mean difference. The middle 95% of those 10,000 means is the **95% confidence interval (CI)**. **If the CI does not include 0, the difference is significant.** “Paired” means both systems are compared on the same users, which removes user-difficulty noise.");
NOTE("Example.", "H1: ARAG+GRU − ARAG = −0.013, CI [−0.032, +0.005]. The CI includes 0 → not significant. H2: ARAG+GRU − GRU = +0.174, CI [+0.069, +0.278]. The CI is entirely above 0 → significant.");

// =====================================================================================
H1("4. The Dataset");
TABLE(["Property", "Value"], [
  ["Source", "Amazon Reviews 2023 (McAuley Lab, Hugging Face `McAuley-Lab/Amazon-Reviews-2023`)"],
  ["Category", "Video_Games"],
  ["Split", "Official rating-only **5-core, leave-last-out** benchmark"],
  ["Users", "94,762 (each has exactly one validation and one test target)"],
  ["Interactions", "814,586 = 625,062 train + 94,762 validation + 94,762 test"],
  ["Items with metadata", "137,269 (only 25,612 = 18.7% ever appear in interactions)"],
  ["Training vocabulary", "25,527 items (items seen in training)"],
  ["History length (train)", "mean 6.6, **median 4**, 75% of users ≤ 7, 63.5% have 3–5"],
  ["Item text", "title + main category + categories + features + description, ≤120 words; **review text never used**"],
  ["Integrity", "SHA-256 hashes of all 4 raw files stored in `results/manifest.json`"],
], [2600, 6426]);
P("**5-core** means every user and item has at least 5 interactions in the full data. Because one item per user is held out for validation and one for test, training histories can be as short as 3.");
IMG("fig08_history_lengths.png", "Figure 1 — Training history length per user. The median user has only four training purchases, which is why an ID-only GRU struggles.", 5.6);

// =====================================================================================
H1("5. The Evaluation Protocol (Why the Comparison Is Fair)");
H2("5.1 Frozen candidate lists");
P("For every user we build **one list of 20 products**: the true next purchase + **19 negatives (decoys)**. The list is shuffled with a fixed seed (42), saved, hashed, and **never changed**. **Every system ranks exactly the same list.**");
P("Why not let each system retrieve its own candidates? Because then a system could lose simply because its retrieval missed the item, and we would be measuring retrieval instead of ranking. Frozen lists isolate **ranking quality**.");
H2("5.2 How the 19 negatives are chosen");
B([
  "Drawn only from the **25,527 items that appear in training** (so every system “knows” them).",
  "Sampled **proportional to popularity** (training interaction count), so decoys are about as popular as real targets.",
  "Excluding anything the user ever interacted with.",
]);
P("Result: **popularity alone cannot identify the target**. The popularity baseline scores 0.109 NDCG@5 — even slightly below random (0.125). That proves the lists have no popularity shortcut.");
H2("5.3 Which users are evaluated");
B([
  "**100 randomly sampled test users** for every system (LLM calls cost money). All significance tests use these same 100 users.",
  "**100 validation users** for tuning θ and α.",
  "Non-LLM systems are **also** evaluated on **all 94,762 test users**, which gives very precise numbers for them.",
]);
H2("5.4 Tuning rules");
P("Every choice (NLI threshold θ, fusion weight α, recency decay d, the GRU’s best epoch) is made **on validation data only**. The test set is touched once, at the end.");

// =====================================================================================
H1("6. Every System Explained");
H2("6.1 ARAG — the agentic LLM recommender");
P("ARAG (Maragheh et al., 2025, Walmart) splits ranking into four LLM agents. Per user it makes **7 calls**:");
TABLE(["Agent", "What it reads", "What it outputs", "Calls"], [
  ["**UUA** – User Understanding", "long-term history (≤50 items) + last 5 items", "interests, current intent, dislikes, ≤120-word summary", "1"],
  ["**NLI** – Natural Language Inference", "raw history + 5 candidates per call", "per candidate: support / neutral / contradict, score 0–10, reason", "4"],
  ["**CSA** – Context Summary", "UUA summary + candidates with score ≥ θ (θ* = 5)", "≤200-word summary of the relevant context", "1"],
  ["**IRA** – Item Ranker", "UUA summary + CSA summary + all 20 candidates", "full ranking of the 20 + reasons for the top 5", "1"],
], [2300, 2600, 3226, 900]);
P("UUA and NLI run **in parallel**; then CSA; then IRA. History is shown to the LLM as lines like `title | category | rating | date`.");
B([
  "**Robustness:** if NLI skips a candidate it is re-asked once, then given a neutral score 5. If no candidate passes θ, CSA uses the top 3. A **ranking parser** repairs broken IRA outputs (unknown IDs dropped, duplicates removed, missing IDs appended). Every repair and fallback is counted.",
  "**Ablations (from the paper):** _ARAG w/o NLI_ — CSA summarises all 20 candidates; _ARAG w/o NLI & CSA_ — IRA gets only the user summary and raw candidates.",
]);
H2("6.2 LLM baselines (from the paper)");
B([
  "**LLM-Recency:** one prompt with the last 10 history items + 20 candidates.",
  "**LLM-Vanilla-RAG:** one prompt with the 10 history items most similar (MiniLM) to the mean candidate embedding.",
]);
H2("6.3 The ID-only GRU (original trained model)");
TABLE(["Setting", "Value"], [
  ["Architecture", "item embedding (64) → dropout → GRU (hidden 128) → dropout → linear projection (128→64)"],
  ["Scoring", "user vector · item embedding (tied input/output embeddings)"],
  ["Vocabulary", "25,527 items + padding (0) + out-of-vocabulary (1)"],
  ["Parameters", "≈ 1.72 M (1.63 M of them are the item embeddings)"],
  ["Training", "last 50 train items per user; next-item at every position; full softmax; Adam lr 1e-3; batch 256"],
  ["Model selection", "grid hidden {64,128} × dropout {0.2,0.5} → best 128/0.2; ≤12 epochs, early stopping (patience 3) on validation NDCG@5 over all 94,762 users; 3 seeds"],
  ["Result", "test NDCG@5 **0.279 ± 0.006** (all users, 3 seeds)"],
], [2200, 6826]);
H2("6.4 The text-aware GRU (our improvement)");
P("**Only one thing changes: how an item is represented.**");
CODE([
  "ID-only GRU:     item vector = e(i)                      (learned, 64-d)",
  "Text-aware GRU:  item vector = e(i) + W · MiniLM(text_i)  (MiniLM frozen 384-d, W learned 384→64)",
]);
B([
  "Same architecture, same hidden size, same dropout, same 12-epoch budget, same 3 seeds → a fair comparison.",
  "The combined vector is used for both input and the tied output, so items with similar descriptions end up close even if they were rarely bought together.",
  "Text exists for 25,526 of 25,527 vocabulary items. Extra parameters: only W = 24,576.",
  "Training: ~25 s per epoch, ~5 min per seed on a laptop CPU. Validation NDCG@5: 0.476 / 0.473 / 0.477 (vs 0.293 for the ID-only GRU).",
  "Result: test NDCG@5 **0.461 ± 0.002** on all 94,762 users.",
]);
IMG("fig11_text_gru_training.png", "Figure 2 — Text-aware GRU validation score per epoch. It passes the ID-only GRU’s best by epoch 3 and is still rising at the 12-epoch cap.", 5.6);
H2("6.5 Non-learned baselines");
B([
  "**Random:** shuffled with a seed. Expected Hit@5 = 0.25.",
  "**Popularity:** order by training purchase count.",
  "**Content similarity:** user profile = recency-weighted sum of the MiniLM vectors of their history (weight e^(−0.3·age)); rank candidates by similarity. No training.",
  "**ItemKNN:** item–item cosine similarity from “bought by the same users” counts; score = recency-weighted sum of similarities to history items.",
]);
H2("6.6 Late fusion (combining two rankings)");
P("The LLM outputs a ranking, not scores, so we convert each ranking into scores and blend:");
CODE([
  "s(i)     = 1 − (rank(i) − 1) / 19          (top item = 1.0, last = 0.0)",
  "final(i) = α · s_LLM(i) + (1 − α) · s_GRU(i)",
]);
P("α = 1 gives pure ARAG, α = 0 gives pure GRU. α is chosen on validation: **0.9** for ARAG+GRU, **0.3** for ARAG+text-GRU, **0.4** for ARAG+content. Ties are broken by NLI score.");

// =====================================================================================
H1("7. Code Walkthrough");
H2("7.1 Repository map");
TABLE(["Path", "What lives there"], [
  ["`configs/config.yaml`", "every path, seed, model name and grid — the single source of settings"],
  ["`configs/prompts/`", "the agent prompts (UUA, NLI, CSA, IRA, Recency, Vanilla-RAG, search)"],
  ["`src/arag/dataprep/`", "download, load, item text, sequences, frozen lists, manifest"],
  ["`src/arag/retrieval/`", "MiniLM embeddings and the FAISS index"],
  ["`src/arag/llm/`", "LLM client (OpenAI/Ollama), disk cache, output schemas"],
  ["`src/arag/agents/`", "UUA, NLI, CSA, IRA, the ARAG orchestration, LLM baselines"],
  ["`src/arag/models/`", "`gru.py` (model, incl. text features), `train_gru.py`, `score.py`"],
  ["`src/arag/ranking/`", "`fusion.py`, `parser.py`, `simple.py` (random/popularity), `content.py` (content, ItemKNN)"],
  ["`src/arag/eval/`", "`run.py` (the whole experiment), `metrics.py`, `bootstrap.py`, `report.py`, `verify.py`"],
  ["`src/arag/memory/`, `src/arag/api/`", "SQLite storage and the FastAPI backend for the demo"],
  ["`frontend/`", "React + Vite demo user interface"],
  ["`tests/`", "pytest suite (offline tests + tests that need Ollama)"],
  ["`results/`", "committed outputs: `tables.md`, `report.md`, `metrics.json`, predictions"],
  ["`docs/`", "REPORT.md, PROCESS.md, OVERVIEW.md, figures, `paper/main.pdf`, this guide"],
  ["`artifacts/` (not in git)", "frozen lists, GRU checkpoints, FAISS index, LLM cache, demo DB"],
], [3300, 5726]);
H2("7.2 Data flow, step by step");
N([
  "`make data` downloads the CSVs and metadata, checks hashes, builds `interactions.parquet` and `items.parquet` (with item text).",
  "`make lists` builds per-user sequences and the frozen 20-item lists; asserts time ordering; records list hashes.",
  "`make index` embeds item text with MiniLM and builds the FAISS index.",
  "`make gru` trains the ID-only GRU (grid + 3 seeds); `make gru-text` trains the text-aware GRU (3 seeds).",
  "`make eval` (`eval/run.py`): tunes θ and α on validation, runs every system on test, fuses, computes metrics and bootstrap CIs, writes `results/metrics.json` and per-user predictions.",
  "`make report` writes `results/tables.md`, `report.md` and the α-curve plot. `make verify` recomputes every metric from raw predictions and checks hashes.",
]);
H2("7.3 Key functions to know");
TABLE(["Function", "File", "Purpose"], [
  ["`GRU4Rec.item_emb`", "models/gru.py", "item vector (ID embedding + projected text if enabled)"],
  ["`GRU4Rec.encode`", "models/gru.py", "user vector from a history"],
  ["`train_model`, `evaluate`", "models/train_gru.py", "training loop with early stopping on validation NDCG@5"],
  ["`fuse`", "ranking/fusion.py", "late fusion of two rankings with weight α"],
  ["`content_ranking`, `ItemKNN`", "ranking/content.py", "the two non-learned baselines"],
  ["`parse_ranking`", "ranking/parser.py", "repairs malformed LLM rankings"],
  ["`run_uua`, `run_nli`, `run_csa`, `run_ira`", "agents/*.py", "the four agents"],
  ["`paired_bootstrap`", "eval/bootstrap.py", "confidence intervals for differences"],
  ["`LLMClient`", "llm/client.py", "API calls, retries, cache, cost tracking, `REPLAY_ONLY`"],
], [3100, 2300, 3626]);
H2("7.4 Reproducibility machinery");
B([
  "**LLM disk cache** keyed by SHA-256 of (model, prompt, schema): re-running gives identical outputs and zero new API calls.",
  "**`REPLAY_ONLY=1`** makes any uncached LLM call raise an error — it is impossible to spend money by accident.",
  "**Run record** (`results/run_record.json`): git commit, config, seeds, model, dataset manifest.",
  "**`make verify`**: recomputes all metrics from the raw per-user rankings and checks list/dataset hashes. Prints `VERIFY OK`.",
]);

// =====================================================================================
H1("8. Running the Whole Pipeline");
H2("8.1 One-time setup");
CODE([
  "make setup          # Python venv + dependencies + frontend npm install",
  "export OPENAI_API_KEY=sk-...   # only needed for NEW LLM calls (never commit it)",
]);
H2("8.2 Full pipeline");
CODE([
  "make data lists index     # dataset -> frozen lists -> MiniLM + FAISS",
  "make gru gru-text         # ID-only GRU (~1h50 CPU) and text-aware GRU (~15 min)",
  "make eval                 # all systems; LLM calls cached on disk",
  "make report verify        # results/ + full audit (expect: VERIFY OK)",
  "python scripts/make_figures.py        # docs/figures/",
  "python scripts/make_paper_tables.py   # LaTeX tables for docs/paper/",
]);
H2("8.3 The zero-cost re-run (use this!)");
CODE(["REPLAY_ONLY=1 OPENAI_API_KEY=replay-only-dummy make eval   # ~5 min, $0"]);
P("Every LLM answer is served from the cache; the dummy key only satisfies the OpenAI client and is never used. This is how all extension results were produced.");
H2("8.4 Tests");
CODE([
  ".venv/bin/python -m pytest -q tests/test_content.py tests/test_gru.py tests/test_fusion.py \\",
  "    tests/test_metrics.py tests/test_bootstrap.py      # offline tests, ~2 s",
  "make test     # full suite; needs a local Ollama server with qwen3:8b",
]);
H2("8.5 Cost summary");
TABLE(["Item", "Cost"], [
  ["Original (superseded) evaluation", "2,405 calls, $0.99"],
  ["Corrected evaluation", "2,263 calls, ≈ $1.06"],
  ["Total project LLM spend", "4,668 calls, 10.3 M input + 0.83 M output tokens, **$2.04**"],
  ["Extension (content baselines, text-aware GRU, new fusions)", "**$0** (replay only)"],
  ["Demo, per uncached feed page", "7 calls ≈ $0.003 with gpt-4o-mini"],
], [5000, 4026]);

// =====================================================================================
H1("9. Testing the System from the Frontend (Choosing Users)");
P("**Yes — the demo lets you pick a user and see live recommendations, the agents’ reasoning, search, feedback and the benchmark results.** The demo uses **8 real test users** from the dataset (no fake users). It now fuses ARAG with the **text-aware GRU** (α = 0.3); set `app.gru_variant: id` in `configs/config.yaml` to demo the original GRU instead.");
H2("9.1 Start the demo");
CODE([
  "# Option A — full live demo (new LLM calls allowed, ~$0.003 per new page)",
  "export OPENAI_API_KEY=sk-...",
  "make app",
  "",
  "# Option B — fully offline (only cached LLM answers; uncached requests show an error)",
  "REPLAY_ONLY=1 OPENAI_API_KEY=replay-only-dummy make app",
  "",
  "# then open  http://127.0.0.1:8000     (different port:  PORT=8001 make app)",
]);
P("The first start takes ~20–60 s (frontend build + loading MiniLM, FAISS and the GRU). Check `http://127.0.0.1:8000/health` — it should show `\"gru_variant\": \"text\"` and `\"alpha\": 0.3`.");
H2("9.2 Step-by-step test script for the team");
N([
  "**Pick a user** in the “Demo user” dropdown at the top. Expand “This user’s recent history” to see what they bought.",
  "**Feed tab:** a personalised list of 10 items per page. Candidates come from FAISS (similar to the user’s history) ∪ GRU top items, then ARAG + text-GRU rank them. Each card shows a reason and evidence. Click “load more” for the next page.",
  "**Open the agent trace** (drawer below the results): see the UUA’s interests/intent, every NLI score (accepted ones highlighted, threshold θ), the CSA summary and the final ranking. This is the best thing to show a professor.",
  "**Give feedback:** Like, Save or Dismiss an item. Dismissed items never appear again for that user; the GRU’s user vector is recomputed from the updated sequence on the next request (no retraining).",
  "**Search tab:** type e.g. “wireless headset under 50 dollars”. The LLM parses the query into keywords, max price and exclusions; FAISS finds matches; ARAG + GRU re-rank them for this user; top 5 are shown with reasons.",
  "**Profile tab:** view, regenerate or delete the stored UUA profile (what the agent “thinks” about this user, with which events it used).",
  "**Benchmarks tab:** the evaluation results — main table, H1/H2 cards with confidence intervals, and the α curve.",
  "**Switch users** and compare: a console-game buyer vs. an accessories buyer should get visibly different feeds.",
]);
NOTE("Offline caveat.", "In replay mode only requests whose LLM answers are already cached work. Pages and queries the team tried before will replay instantly; new ones need Option A (an API key). Verified on this machine: health, users, feed pages and a search all work in replay mode.", "FFF4E5", "C55A11");
H2("9.3 What to say while demoing");
B([
  "“Every card is ranked by four LLM agents plus our trained text-aware GRU, blended with α tuned on validation.”",
  "“This trace shows the NLI agent scoring each candidate against the history — only those with score ≥ 5 reach the summary agent.”",
  "“Dismiss is permanent and immediately changes the next recommendations.”",
  "“The benchmark numbers on this tab are the same audited numbers as in our report.”",
]);

// =====================================================================================
H1("10. Results and How to Read Them");
H2("10.1 All systems on the same 100 test users (NDCG@5)");
TABLE(["System", "Hit@5", "NDCG@5", "Group"], [
  ["Random", "0.220", "0.125", "trivial"],
  ["Popularity", "0.200", "0.109", "trivial"],
  ["GRU, ID only (seed 0)", "0.390", "0.238", "trained"],
  ["LLM-Vanilla-RAG", "0.380", "0.263", "LLM baseline"],
  ["LLM-Recency", "0.400", "0.285", "LLM baseline"],
  ["ARAG w/o NLI & CSA", "0.470", "0.362", "ablation"],
  ["ARAG w/o NLI", "0.510", "0.376", "ablation"],
  ["ItemKNN", "0.540", "0.399", "non-learned"],
  ["ARAG + GRU (α=0.9)", "0.540", "0.412", "fusion"],
  ["ARAG + content (α=0.4)", "0.590", "0.423", "fusion"],
  ["**ARAG**", "0.530", "**0.424**", "agentic LLM"],
  ["Content similarity", "0.590", "0.439", "non-learned"],
  ["**Text-aware GRU (seed 0)**", "**0.660**", "**0.462**", "trained (ours)"],
  ["**ARAG + text-aware GRU (α=0.3)**", "**0.660**", "**0.468**", "fusion (ours)"],
], [3600, 1500, 1500, 2426]);
H2("10.2 Non-LLM systems on all 94,762 test users");
TABLE(["System", "Hit@5", "NDCG@5", "MRR"], [
  ["Random", "0.252", "0.149", "0.181"],
  ["Popularity", "0.213", "0.128", "0.166"],
  ["GRU, ID only (3 seeds)", "0.422 ± 0.008", "0.279 ± 0.006", "0.285 ± 0.004"],
  ["ItemKNN", "0.519", "0.406", "0.406"],
  ["Content similarity", "0.569", "0.411", "0.402"],
  ["**Text-aware GRU (3 seeds)**", "**0.652 ± 0.002**", "**0.461 ± 0.002**", "**0.435 ± 0.002**"],
], [3300, 1900, 1900, 1926]);
IMG("fig10_content_aware.png", "Figure 3 — The extension. Left: all 94,762 users. Right: the same 100 users as the LLM systems.", 6.3);
H2("10.3 Significance (paired bootstrap, ΔNDCG@5)");
TABLE(["Contrast", "Δ", "95% CI", "Significant?"], [
  ["H1: ARAG+GRU − ARAG", "−0.013", "[−0.032, +0.005]", "no"],
  ["H2: ARAG+GRU − GRU", "+0.174", "[+0.069, +0.278]", "**yes**"],
  ["ARAG − LLM-Recency", "+0.139", "[+0.063, +0.217]", "**yes**"],
  ["ARAG − LLM-Vanilla-RAG", "+0.162", "[+0.079, +0.247]", "**yes**"],
  ["ARAG − ARAG w/o NLI & CSA", "+0.062", "[+0.016, +0.113]", "**yes**"],
  ["ARAG − ARAG w/o NLI", "+0.049", "[−0.011, +0.111]", "no"],
  ["Text-GRU − GRU (100 users)", "+0.224", "[+0.156, +0.294]", "**yes**"],
  ["Text-GRU − GRU (all 94,762 users)", "+0.174", "[+0.172, +0.176]", "**yes**"],
  ["Text-GRU − content (all users)", "+0.050", "[+0.047, +0.053]", "**yes**"],
  ["Text-GRU − ARAG (100 users)", "+0.038", "[−0.067, +0.144]", "no"],
  ["ARAG+text-GRU − ARAG", "+0.044", "[−0.035, +0.122]", "no"],
], [3700, 1300, 2300, 1726]);
IMG("fig03_hypotheses.png", "Figure 4 — Forest plot of the original study’s key contrasts. A bar crossing zero means “not significant”.", 5.6);
IMG("fig05_alpha_curve.png", "Figure 5 — ARAG + ID-only GRU vs the fusion weight α. Both curves rise towards α = 1 (pure ARAG): the ID-only GRU adds nothing.", 5.0);
H2("10.4 How to interpret the results in one minute");
N([
  "ARAG is a genuinely strong LLM system and the paper’s claims reproduce.",
  "The ID-only GRU was weak because histories are short; fusing it into ARAG did not help.",
  "Most of ARAG’s advantage over the ID-only GRU came from **reading item text**, not from agentic reasoning: a zero-training content-similarity ranker ties with ARAG.",
  "Giving the GRU item text makes it the best single model on all users and at least as good as ARAG — at no API cost.",
  "With only 100 LLM-evaluated users, we cannot separate the top systems from each other; we say so honestly.",
]);

// =====================================================================================
H1("11. Comparison with the Original ARAG Paper");
P("Paper: Maragheh et al., “ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation”, arXiv:2506.21931 (2025). They used Amazon 2016 data (Clothing, Electronics, Home), gpt-3.5-turbo and 10,000 users, without significance tests. We used Amazon 2023 Video Games, gpt-4o-mini, 100 users, with paired bootstrap. **Absolute numbers are not comparable; claims are.**");
TABLE(["System (NDCG@5)", "Paper: Clothing", "Paper: Electronics", "Paper: Home", "Ours: Video Games"], [
  ["Recency", "0.309", "0.225", "0.224", "0.285"],
  ["Vanilla RAG", "0.299", "0.238", "0.229", "0.263"],
  ["ARAG w/o NLI & CSA", "0.302", "0.272", "0.249", "0.362"],
  ["ARAG w/o NLI", "0.385", "0.296", "0.273", "0.376"],
  ["ARAG", "0.439", "0.329", "0.289", "0.424"],
], [2600, 1600, 1700, 1500, 1626]);
P("The ablation “staircase” (w/o NLI&CSA < w/o NLI < full ARAG) holds in all four columns. ARAG’s lift over the best baseline is +48.8% for us vs +25.6% to +42.1% in the paper.");
IMG("fig04_ablation.png", "Figure 6 — Ablation staircase: ours next to the paper’s three categories.", 5.6);

// =====================================================================================
H1("12. The Mistake We Found and Fixed (Errata)");
P("**Be ready to tell this story — professors love it, and it shows scientific honesty.**");
H2("12.1 What went wrong");
P("The first version sampled the 19 negatives **uniformly from all 137,260 catalogue items**. But 81% of those items have **zero** interactions. On average **15.5 of 19 negatives were unknown** to Popularity (all tied at 0) and to the GRU (all mapped to the same “unknown” embedding). Those systems effectively faced a 4.5-way choice instead of a 20-way one. Popularity reached Hit@5 = 0.97, the GRU NDCG@5 = 0.82, and fusion appeared to improve ARAG by +0.349 — a **false “H1 supported”** result.");
H2("12.2 How it was fixed");
B([
  "Negatives re-sampled from the 25,527 known items, proportional to popularity (Section 5.2).",
  "Every system re-evaluated; the GRU retrained from scratch; all systems compared on the same 100 users.",
  "The original numbers are kept (not deleted) in REPORT.md Appendix E and the paper’s Appendix B.",
]);
TABLE(["Quantity", "Flawed lists", "Corrected lists"], [
  ["Unknown negatives per list", "15.5 / 19", "0 / 19"],
  ["Popularity NDCG@5", "0.801", "0.109"],
  ["GRU NDCG@5 (all users)", "0.823", "0.279"],
  ["H1 (fusion − ARAG)", "+0.349 “significant”", "−0.013 not significant"],
  ["Best system", "GRU / fused", "ARAG (then text-aware GRU)"],
], [3600, 2700, 2726]);
IMG("fig07_protocol_flaw.png", "Figure 7 — The flawed lists vs the corrected lists. The conclusion reversed.", 6.0);
NOTE("Lesson.", "How you build the candidate list can create or erase a whole conclusion. Always check that no system can win through a shortcut the others cannot use (see Krichene & Rendle, KDD 2020, on sampled metrics).");

// =====================================================================================
H1("13. Limitations and Future Work");
TABLE(["Limitation", "Why it matters / what we would do"], [
  ["Only 100 LLM-evaluated users", "CIs ≈ ±0.08–0.10; top systems cannot be separated. ~300 users (~$3) would resolve ~0.05 differences."],
  ["Sampled negatives", "Easier than real retrieval candidates; results measure ranking on hard 20-item lists, not end-to-end retrieval."],
  ["Negatives slightly more popular than targets", "Popularity scores slightly below random; slightly disadvantages frequency-sensitive models (incl. GRUs). Effect ~0.04 Hit@5, smaller than our key gaps."],
  ["Leave-last-out leakage", "Other users’ later purchases can be in GRU/ItemKNN training; inherent to the official split, equal across those systems."],
  ["Text-GRU still improving at 12 epochs", "Cap kept for fairness with the ID-only GRU; our gain is conservative. Future: longer training for both."],
  ["One category, one LLM", "Results may not transfer; future: more categories, other LLMs."],
  ["Offline metrics only", "No user study or A/B test."],
], [3200, 5826]);
P("**Future work:** SASRec (Transformer) with the same text-aware item vectors; feed the GRU’s scores to the IRA agent as a feature instead of fusing afterwards; retriever-mined negatives; more users.");

// =====================================================================================
H1("14. Professor Q&A Bank");
P("Answers are written to be said out loud. Numbers are exact; memorise the ones in bold.");

H2("14.1 Motivation and scope");
QA("What problem does your project solve?", "Next-item recommendation: given a user’s purchase history on Amazon (Video Games), rank 20 candidate products so the one they actually bought next is near the top. We compare an agentic LLM recommender (ARAG) with a trained deep sequence model (GRU), and their combination.");
QA("Why is this a deep-learning project if you use GPT?", "The trained component is ours: a PyTorch GRU (≈1.7 M parameters) trained with a full-softmax next-item objective, plus our text-aware extension that adds a learned projection of frozen transformer (MiniLM) embeddings. The LLM is a strong baseline and fusion partner; the deep-learning contribution and the main improvement (0.279 → 0.461) are in the trained model.");
QA("What was your hypothesis?", "H1: fusing the GRU into ARAG beats ARAG alone. H2: the fused system beats the GRU alone. **H1 failed (−0.013, n.s.), H2 held (+0.174, significant).**");
QA("Is this a reproduction of the ARAG paper?", "It is ARAG-inspired, not an exact reproduction: the paper did not publish prompts, its threshold or candidate construction; it used gpt-3.5-turbo on 2016 data. We test whether its claims hold, and they do: ARAG beats Recency and Vanilla-RAG significantly and the NLI+CSA ablation hurts significantly.");
QA("What is new in your work compared to the paper?", ["(1) Fusion with a trained GRU. (2) Statistical testing (paired bootstrap) on every contrast. (3) Non-LLM baselines including popularity, content and ItemKNN. (4) The text-aware GRU that matches ARAG at zero API cost. (5) A documented protocol flaw and fix."]);

H2("14.2 Data");
QA("Which dataset and why?", "Amazon Reviews 2023, Video Games, official 5-core leave-last-out split: real data, an official split (no custom leakage), rich item metadata for the LLM, and a category the ARAG paper never ran.");
QA("How big is it?", "**94,762 users, 814,586 interactions, 25,527 items in the training vocabulary.** Median training history: **4 items**.");
QA("What does 5-core mean?", "Every user and item has at least 5 interactions. After holding out the validation and test items, a user can have as few as 3 training items.");
QA("Do you use review text?", "No — only product metadata (title, categories, features, description, ≤120 words). Using reviews could leak the target purchase.");
QA("How do you prevent data leakage in the split?", "We use the official leave-last-out files and assert max(train time) ≤ validation time ≤ test time, and that targets never appear in their own history; the pipeline fails hard otherwise. One known caveat: other users’ later purchases can appear in training (inherent to this split), affecting the trained models equally.");

H2("14.3 Evaluation protocol");
QA("Why do all systems rank the same 20 candidates?", "To measure ranking, not retrieval. If each system retrieved its own candidates, a retrieval miss would count as a ranking failure.");
QA("How are the 19 negatives chosen and why?", "From the 25,527 items seen in training, proportional to popularity, excluding the user’s own items. This makes decoys about as popular as targets, so popularity alone gives no advantage — popularity scores 0.109, below random (0.125).");
QA("Why only 100 users for the LLM systems?", "Cost and time: ARAG makes 7 calls per user. 100 users keeps each run around $1. All systems are compared on the same 100 users, and non-LLM systems are additionally evaluated on all 94,762 users.");
QA("Isn’t 100 users too few?", "It is enough to detect large effects (ARAG vs the ID-only GRU, the baselines, the ablation) but not small ones — the CI is about ±0.08–0.10. That is why we report CIs and say the top systems are statistically tied. ~300 users would resolve ~0.05.");
QA("Why NDCG@5 as the main metric?", "It rewards putting the true item high, not just in the top 5, and it is the paper’s headline metric, so our results are comparable in kind.");
QA("What is the expected Hit@5 of random ranking?", "5/20 = **0.25**. Our random baseline scores 0.252 on all users (0.220 on the 100-user sample, within noise) — a sanity check that the evaluation is correct.");
QA("Explain the paired bootstrap.", "For each user, compute the NDCG difference between two systems. Resample the users with replacement 10,000 times and take the mean difference each time. The 2.5th–97.5th percentiles form the 95% CI. If it excludes zero, the difference is significant. Pairing removes the variation in how hard each user is.");
QA("Why not a t-test?", "Per-user NDCG is highly non-normal (many zeros). The bootstrap makes no normality assumption. A Wilcoxon signed-rank test would be a valid alternative.");
QA("How did you tune hyperparameters without touching test?", "θ, α, the recency decay and the GRU checkpoint were all chosen on validation users; the test set was evaluated once at the end.");

H2("14.4 ARAG and the LLM side");
QA("Explain the four ARAG agents.", "UUA summarises the user’s interests and intent; NLI scores every candidate 0–10 for compatibility with the history; CSA summarises the candidates that pass θ = 5; IRA ranks all 20 using both summaries. UUA and NLI run in parallel. 7 LLM calls per user.");
QA("What is θ and how was it chosen?", "The NLI acceptance threshold. Validation NDCG@5 for θ = 4/5/6/7 was 0.4228/**0.4305**/0.3993/0.3962, so θ* = 5.");
QA("What if the LLM returns invalid output?", "Outputs use strict JSON schemas and are validated. One retry with the validation error; then a documented fallback (e.g. neutral NLI score 5). The ranking parser repairs unknown or duplicate IDs. Every fallback and repair is counted, not hidden.");
QA("How do you make LLM results reproducible?", "Temperature 0, pinned model name, and a disk cache keyed by the hash of (model, prompt, schema). Re-running replays identical answers at zero cost; `REPLAY_ONLY=1` forbids any new call.");
QA("Why gpt-4o-mini?", "Cheap (≈$0.15/$0.60 per million tokens), fast and supports strict JSON schemas. The whole project cost **$2.04**. The client also supports a local Ollama model (qwen3:8b) for zero-cost testing.");
QA("Which ablation matters most?", "Removing NLI and CSA together significantly hurts (−0.062, CI [−0.113, −0.016]). Removing only NLI is directionally negative but not significant (−0.049).");

H2("14.5 The GRU and the text-aware GRU");
QA("Describe your GRU architecture.", "Item embedding (64) → dropout → 1-layer GRU (hidden 128) → dropout → linear projection to 64; the score is the dot product with tied item embeddings; full-softmax cross-entropy over 25,527 items at every position; Adam, lr 1e-3, batch 256, early stopping on validation NDCG@5.");
QA("Why a GRU and not an LSTM or Transformer?", "GRU4Rec is the classic deep sequential recommender, with fewer parameters than an LSTM and easy to train on CPU. With median history 4, long-range modelling (the Transformer’s strength) matters little. SASRec with text features is natural future work.");
QA("Why did the original GRU perform poorly?", "Short histories (median 4) and sparse item IDs: most items appear in few sequences, so their embeddings are poorly learned; the GRU cannot read item content. An untrained content-similarity ranker (0.411) beats it (0.279).");
QA("What exactly is the text-aware GRU?", "Same GRU, but each item vector is e(i) + W·MiniLM(text_i): a learned ID embedding plus a learned 384→64 projection of the item’s frozen MiniLM text embedding, used for both input and tied output. Only 24,576 extra parameters.");
QA("Why does adding text help so much?", "It lets the model generalise across items: two products with similar descriptions get similar vectors even if they were rarely bought together. With 3–5 purchases per user, this shared content signal fills in what the ID embeddings cannot learn.");
QA("Is the comparison between the two GRUs fair?", "Yes: same architecture, hidden size, dropout, optimiser, 12-epoch budget, early-stopping rule and seeds. The only difference is the item representation.");
QA("Did the text-aware GRU overfit?", "Validation NDCG@5 rose every epoch up to the cap (0.476), and test (0.461) is close to validation, across 3 seeds with spread ±0.002. No sign of overfitting; if anything it was under-trained.");
QA("Why freeze MiniLM?", "Cheaper and safer: the text encoder is pre-trained on general language; fine-tuning it on 625k short sequences on a CPU risks overfitting and costs much more. The learned projection W adapts it to our task.");
QA("How many parameters?", "ID-only GRU ≈ **1.72 M** (1.63 M in the item embedding table). Text-aware GRU ≈ **1.74 M** (+24,576 for W).");
QA("How long does training take?", "ID-only grid + 3 seeds ≈ 1 h 50 min on CPU. The text-aware GRU ≈ 5 minutes per seed (~25 s/epoch) after our memory optimisations.");
QA("What memory problems did you hit?", "Training was killed for running out of memory on 16 GB. Two fixes: compute logits only at real (non-padded) positions, and encode validation users in chunks of 4,096 with history capped at 50. Same loss; the original GRU’s validation score was reproduced exactly (0.2928) as a check.");

H2("14.6 Fusion");
QA("How do you combine the LLM and the GRU?", "Late fusion of rankings: convert each rank to a score 1 − (rank−1)/19 and blend final = α·LLM + (1−α)·GRU. α = 1 is pure ARAG, α = 0 is pure GRU; α is tuned on validation.");
QA("Why late fusion instead of feeding the GRU into the LLM?", "The LLM outputs a permutation, not scores, and late fusion makes the GRU’s contribution exactly ablatable (α = 1 recovers ARAG). Giving GRU scores to the ranker agent as a feature is listed as future work.");
QA("Why did ARAG + GRU not help?", "The validation-optimal α was 0.9 (90% LLM) and the test curve peaks at α = 1. The ID-only GRU’s signal is weaker than, and largely contained in, what the LLM already uses.");
QA("Did fusion help with the text-aware GRU?", "ARAG + text-GRU reaches the highest 100-user score (0.468, α = 0.3), but it is not significantly better than either ARAG (+0.044) or the text-GRU alone (+0.006).");
QA("Why is α = 0.3 for the text-GRU but 0.9 for the ID-GRU?", "Validation decides how much each signal is worth: the text-aware GRU is strong, so it gets 70% weight; the ID-only GRU was weak, so it got 10%.");

H2("14.7 Results and interpretation");
QA("What is your best system?", "On all 94,762 users the text-aware GRU (0.461 NDCG@5). On the 100-user sample ARAG + text-GRU is highest (0.468), but it is statistically tied with ARAG (0.424), the text-GRU (0.462) and content similarity (0.439).");
QA("So is the LLM useless?", "No. Among LLM methods, the agentic design clearly matters (ARAG beats Recency by +0.139 and Vanilla-RAG by +0.162, significant). But on this task a trained model with item text is at least as good and much cheaper.");
QA("Why does popularity score below random?", "Negatives are sampled proportional to popularity and end up on average slightly more popular than targets (1.01× validation, 1.40× test). Ranking by popularity therefore pushes decoys up slightly. It confirms the lists remove the popularity shortcut.");
QA("How confident are you in the main result?", "Very, for the text-GRU vs ID-GRU gain: +0.174 with CI [+0.172, +0.176] on 94,762 users, consistent across 3 seeds. Less for the ARAG comparisons, where n = 100 gives wide CIs, which we state explicitly.");
QA("What is the cost–quality trade-off?", "ARAG: 7 API calls per user, seconds of latency. Text-aware GRU: scores all 94,762 users in seconds on a CPU, about 15 minutes to train 3 seeds, $0 per request — and at least as accurate here.");

H2("14.8 Engineering, reproducibility and ethics");
QA("How can we verify your numbers?", "`make verify` recomputes every metric from the saved per-user rankings, checks dataset and list hashes against the run record, and prints VERIFY OK. All tables in the paper are generated from `results/metrics.json` by a script.");
QA("How much did it cost?", "**$2.04** in total LLM spend (4,668 calls). All extension experiments cost $0 thanks to the cache.");
QA("Tell us about a mistake you made.", "Our first candidate lists drew negatives from the whole catalogue, 81% of which had no interactions, so popularity and the GRU could spot the target easily and we wrongly concluded H1 was supported (+0.349). We found it in review, rebuilt the lists, re-ran everything and kept the old numbers in an errata appendix.");
QA("What tests do you have?", "A pytest suite: metrics, bootstrap, fusion (α = 0/1 reproduce each input exactly), the parser, the lists, GRU padding invariance and save/load, the text-feature model, content/ItemKNN, the API and the agents (the last two use a local Ollama model).");
QA("Any privacy or ethical concerns?", "The data is public and pseudonymised (hashed user IDs). LLM prompts contain only product metadata and anonymised histories. Recommenders can reinforce popularity bias; our protocol explicitly removes the popularity shortcut in evaluation.");

H2("14.9 Curve-ball questions");
QA("If you had one more week, what would you do?", "Evaluate 300+ users with the LLM to separate the top systems, train SASRec with the same text-aware item vectors, and try giving the GRU’s scores to the ranker agent directly.");
QA("Would the results hold with a better LLM (e.g. GPT-4o)?", "Possibly ARAG would improve; but the content-similarity result suggests much of the gain comes from reading item text, which a trained model can also do. It is an open question we would test.");
QA("Why not just use content similarity, since it ties with ARAG?", "Content similarity ignores collaborative patterns. The text-aware GRU combines both and significantly beats content similarity on all users (+0.050, CI [+0.047, +0.053]).");
QA("Is NDCG@5 on 20 sampled candidates realistic?", "It is a controlled ranking benchmark. Real systems rank hundreds of retrieved items; sampled metrics can distort rankings (Krichene & Rendle, 2020). We acknowledge this as a limitation and designed the negatives to be hard.");
QA("Could the text-aware GRU be cheating by seeing the target’s text?", "No. It sees the text of every vocabulary item, exactly as ARAG sees every candidate’s text; it never sees which candidate is the target, and no review text or held-out interactions are used.");

// =====================================================================================
H1("15. Glossary");
TABLE(["Term", "Meaning"], [
  ["ARAG", "Agentic Retrieval-Augmented Generation recommender: UUA, NLI, CSA, IRA agents"],
  ["UUA / NLI / CSA / IRA", "User Understanding / Natural Language Inference / Context Summary / Item Ranker agents"],
  ["GRU", "Gated Recurrent Unit, a recurrent neural network for sequences"],
  ["GRU4Rec", "Hidasi et al. 2016: GRU for session-based recommendation"],
  ["Text-aware GRU", "our GRU whose item vectors = ID embedding + projected MiniLM text embedding"],
  ["MiniLM", "small pre-trained sentence encoder producing 384-d text embeddings"],
  ["FAISS", "library for fast nearest-vector search"],
  ["Leave-last-out", "last item = test, second-to-last = validation, rest = train"],
  ["5-core", "users and items with at least 5 interactions"],
  ["Frozen candidate list", "the fixed 20 items (1 target + 19 negatives) every system ranks"],
  ["Negative / decoy", "a candidate that is not the target"],
  ["OOV", "out-of-vocabulary: an item the GRU never saw in training"],
  ["Hit@K / NDCG@K / MRR", "ranking metrics (Section 3.9)"],
  ["Paired bootstrap / CI", "resampling test giving a 95% confidence interval for a difference"],
  ["θ (theta)", "NLI acceptance threshold (θ* = 5)"],
  ["α (alpha)", "fusion weight on the LLM ranking"],
  ["Late fusion", "blending two rankings after each system has ranked"],
  ["Ablation", "removing a component to measure its contribution"],
  ["REPLAY_ONLY", "mode that serves LLM answers only from the cache (zero cost)"],
], [2600, 6426]);

// =====================================================================================
H1("16. Team Study Plan");
TABLE(["Member focus", "Must master", "Sections"], [
  ["Data & protocol", "dataset stats, frozen lists, negatives, metrics, bootstrap, the errata story", "4, 5, 3.9–3.10, 12"],
  ["LLM side", "ARAG agents, θ, prompts, baselines, ablations, paper comparison, cost", "6.1–6.2, 11, 8.5"],
  ["Deep-learning side", "GRU maths, training, text-aware GRU, memory fixes, results", "3.4–3.6, 6.3–6.4, 10"],
  ["Demo & engineering", "running the pipeline, replay mode, frontend demo, tests, verify", "7, 8, 9"],
], [2300, 4426, 2300]);
P("**Everyone** should be able to answer Q1–Q10 and tell the errata story. Practise by having one person read random questions from Section 14 aloud while another answers without looking.");
NOTE("Final tip.", "When unsure, fall back on the protocol: “every system ranks the same 20 items, tuning is validation-only, and every number is re-derivable with make verify.” It is true and it answers most doubts.");

// =====================================================================================
const doc = new Document({
  creator: "ARAG + GRU project team",
  title: "ARAG + GRU Recommender — Complete Project Guide",
  description: "Learning guide, demo instructions and viva Q&A",
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
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "ARAG + GRU — Complete Project Guide", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18, color: "888888" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "docs", "PROJECT_GUIDE.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes;", qn, "questions");
});
