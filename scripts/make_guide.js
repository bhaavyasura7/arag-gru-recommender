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
P("We built a recommender system for video-game products from Amazon. Given the products a person reviewed before, it must guess which product they review next (reviews are the standard stand-in for purchases). We compared two very different ways of doing this. The **first** is an **LLM agent system (ARAG)**: GPT reads the person’s history like a human would and ranks the options, using four cooperating “agents”. The **second** is a **trained neural network (a GRU)** that learns patterns from 625,000 past interactions (reviews). Our question was: **does adding the trained GRU to the LLM system make it better?**");
P("**Yes — once the GRU is trained properly.** Our first GRU looked weak (0.279 NDCG@5) and we blamed short user histories. Later, while adding a second neural baseline (SASRec), we found the real cause: a **weight-initialisation bug** that stopped the GRU from learning. After the fix the plain GRU scores **0.505** on all 94,762 users — better than ARAG (0.424 on its 100 users). Adding the GRU to ARAG now **significantly improves ARAG** (+0.098), but the blend is only as good as the GRU alone. Our **text-aware GRU** (the GRU also reads product descriptions) is the best single model (**0.530**), and a standard Transformer recommender (SASRec, 0.508) confirms the trained-model numbers. ARAG still clearly beats the paper’s own LLM baselines, explains every recommendation in words and has far less popularity bias — but on pure ranking accuracy, cheap trained models win on this data.");
H2("2.2 The three research questions");
TABLE(["Question", "Short answer"], [
  ["RQ1: Does fusing a trained GRU into ARAG improve ranking (H1)? Does ARAG add over the GRU (H2)?", "H1 **yes** (+0.098, significant). H2 **no** (+0.004, not significant): the blend is no better than the GRU alone."],
  ["RQ2: Do the original ARAG paper’s claims hold on our data?", "**Yes.** ARAG beats the paper’s simpler LLM baselines, and removing its NLI+CSA agents significantly hurts. But the paper never compared against trained sequence models, and those beat ARAG here."],
  ["RQ3: How good can trained models get without spending on the LLM?", "Very good: GRU **0.505**, SASRec **0.508**, text-aware GRU **0.530** NDCG@5 on all users, at **$0** per request."],
], [4500, 4526]);
H2("2.3 The five things to remember");
N([
  "**Fair comparison:** every system ranks the *same* 20 candidate products per user (1 real next item + 19 decoys).",
  "**ARAG is a strong LLM system**: NDCG@5 0.424, significantly better than the paper’s simpler LLM baselines.",
  "**Properly trained sequence models are stronger**: GRU 0.520, SASRec 0.536, text-GRU 0.501 on the same 100 users; ARAG vs SASRec is a significant −0.112.",
  "**Fusion helps ARAG** (+0.098, significant) — the LLM and the GRU make different mistakes — but does not beat the GRU alone.",
  "**We caught and fixed two of our own mistakes** — unfair candidate lists, then a GRU initialisation bug — and documented both openly (Section 12).",
]);
NOTE("How to say it in the viva.", "“The paper’s claim holds — ARAG beats the LLM baselines it was compared with. What the paper did not test is a properly trained sequence model; on our data those are more accurate and essentially free, while ARAG’s strengths are explanations, natural-language search and lower popularity bias.”");

// =====================================================================================
H1("3. Background Concepts from Scratch");
H2("3.1 Recommender systems");
P("A recommender system ranks items (products, films, songs) for a user. **Collaborative filtering** uses patterns across many users (“people who bought X also bought Y”). **Content-based** methods use item descriptions (“this is also a racing game for Wii U”). **Sequential recommendation** cares about _order_: what you interacted with most recently matters most.");
H2("3.2 The next-item prediction task");
P("We use **leave-last-out**: for each user, their **last** reviewed item is hidden as the test answer, the **second-to-last** is hidden for validation (tuning), and everything earlier is training history. The system sees the history and must rank candidates so the hidden item comes as high as possible.");
H2("3.3 Embeddings");
P("An **embedding** is a list of numbers (a vector) that represents something. Similar things get similar vectors. Two kinds appear in this project:");
B([
  "**ID embeddings** — learned by the GRU during training, one 64-number vector per product ID. They only become good if the product appears in many training sequences.",
  "**Text embeddings** — produced by **MiniLM** (`all-MiniLM-L6-v2`), a pre-trained sentence encoder that turns a product’s title/category/description into a 384-number vector. Products with similar descriptions get similar vectors even if no user ever reviewed both.",
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
P("**Why initialisation matters with weight tying.** PyTorch’s default embedding initialisation draws every number from N(0, 1). With 64 numbers per item, a dot product between two such vectors is around ±8, so at the start the softmax over 25,527 items is extremely peaked and almost random — the starting loss is about 30 instead of the ideal ≈ ln(25,527) ≈ 10. The gradients are then dominated by fixing this bad start and the model gets stuck. Starting the embeddings small, N(0, 0.02) as in SASRec and BERT, gives a near-uniform softmax and normal training. **This was our GRU bug (Section 12.3).**");
NOTE("How to say it in the viva.", "“With tied embeddings the initial logits are a dot product of two random embeddings; with standard-normal init they are huge, the softmax saturates and training stalls. Small init fixed it: validation NDCG@5 went from 0.29 to 0.53.”");
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
NOTE("Example.", "H1: ARAG+GRU − ARAG = +0.098, CI [+0.013, +0.182]. The CI is entirely above 0 → significant. H2: ARAG+GRU − GRU = +0.004, CI [−0.038, +0.045]. The CI includes 0 → not significant.");
P("For the **94,762-user** comparisons we use an exact shortcut: a user’s NDCG difference can only take a few dozen values (the target is at rank 1…20), so drawing a bootstrap sample is the same as drawing how many users get each value (a multinomial draw). Same answer as resampling users, but fast. (An earlier version bootstrapped a random 20,000-user subset, which made those intervals about twice too wide and slightly off-centre; fixed — no 100-user number changed.)");

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
NOTE("Reviews, not purchases.", "The dataset records items a user **reviewed** (with a star rating), not every purchase — Amazon does not release purchase logs. A user may have bought many more products but reviewed only five. All research on this dataset (including the ARAG paper) treats reviews as the stand-in for purchases; in this guide “history” and “interactions” mean reviewed items.", "FFF4E5", "C55A11");
P("**5-core** means every user and item has at least 5 interactions in the full data. Because one item per user is held out for validation and one for test, training histories can be as short as 3.");
IMG("fig08_history_lengths.png", "Figure 1 — Training history length per user. The median user has only four training interactions. This is normal for Amazon data — and, as Section 10.5 shows, the trained models do just as well for these short-history users.", 5.6);

// =====================================================================================
H1("5. The Evaluation Protocol (Why the Comparison Is Fair)");
H2("5.1 Frozen candidate lists");
P("For every user we build **one list of 20 products**: the true next item + **19 negatives (decoys)**. The list is shuffled with a fixed seed (42), saved, hashed, and **never changed**. **Every system ranks exactly the same list.**");
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
H2("6.3 The ID-only GRU (the trained model)");
TABLE(["Setting", "Value"], [
  ["Architecture", "item embedding (64) → dropout → GRU (hidden 64) → dropout → linear projection (64→64)"],
  ["Initialisation", "item embeddings N(0, 0.02), padding row zero (the fix — see 3.6 and 12.3)"],
  ["Scoring", "user vector · item embedding (tied input/output embeddings)"],
  ["Vocabulary", "25,527 items + padding (0) + out-of-vocabulary (1)"],
  ["Parameters", "≈ 1.66 M (1.63 M of them are the item embeddings)"],
  ["Training", "last 50 train items per user; next-item at every position; full softmax; Adam lr 1e-3; batch 256"],
  ["Model selection", "grid hidden {64,128} × dropout {0.2,0.5} → best 64/0.2 (val 0.531); ≤30 epochs, early stopping (patience 3) on validation NDCG@5 over all 94,762 users; 3 seeds"],
  ["Result", "test NDCG@5 **0.505 ± 0.001** (all users, 3 seeds); before the fix it was 0.279"],
], [2200, 6826]);
P("Training the grid + 3 seeds took about 2 h 25 min on a laptop CPU. The validation score was still creeping up at epoch 30 (0.528 → 0.529), so the GRU is, if anything, slightly under-trained.");
H2("6.4 The text-aware GRU (our improvement)");
P("**Only one thing changes: how an item is represented.**");
CODE([
  "ID-only GRU:     item vector = e(i)                      (learned, 64-d)",
  "Text-aware GRU:  item vector = e(i) + W · MiniLM(text_i)  (MiniLM frozen 384-d, W learned 384→64)",
]);
B([
  "Same training code, optimiser, early-stopping rule, epoch budget and 3 seeds as the ID-only GRU; hidden size 128 / dropout 0.2 (the ID-only GRU’s best setting from the first run, kept fixed to avoid another grid).",
  "The combined vector is used for both input and the tied output, so items with similar descriptions end up close even if they rarely co-occur in histories.",
  "Text exists for 25,526 of 25,527 vocabulary items. Extra parameters for the text projection: only W = 24,576 (total ≈ 1.74 M).",
  "Training: ~40 s per epoch, ~8–9 min per seed on a laptop CPU; early stopping after ~13 epochs. Validation NDCG@5: 0.550 / 0.547 / 0.547.",
  "Result: test NDCG@5 **0.530 ± 0.001** on all 94,762 users — **+0.026** over the ID-only GRU (CI [+0.024, +0.027]), in every history-length group.",
]);
IMG("fig11_text_gru_training.png", "Figure 2 — Validation score per epoch. The text-aware GRU learns much faster (0.53 by epoch 4) and ends higher; the fixed ID-only GRU climbs steadily to 0.53 by epoch 30. The grey line is the old, broken ID-only GRU.", 5.6);
NOTE("How to say it in the viva.", "“Text mainly helps the model learn faster and generalise to rarely-seen items; once the ID-only GRU is trained properly the gap is +0.026, smaller than we first thought, but significant on 94,762 users.”");
H2("6.5 SASRec — a Transformer baseline (added later)");
P("**SASRec** (Kang & McAuley, 2018) is the standard Transformer model for sequential recommendation. Instead of a GRU reading items one by one, **self-attention** lets each position look at all earlier items at once and decide which ones matter. We added it as an independent check on the GRU: if two different architectures give similar numbers, the numbers are trustworthy.");
TABLE(["Setting", "Value"], [
  ["Architecture", "item embedding (64) + learned position embedding → 2 Transformer blocks (2 heads, pre-norm, causal mask) → LayerNorm"],
  ["Same as the GRU", "vocabulary, training sequences, tied full-softmax loss, Adam lr 1e-3, batch 256, ≤30 epochs, early stopping on validation, 3 seeds, frozen test lists"],
  ["Grid", "dropout {0.2, 0.5} → best 0.5 (val 0.530)"],
  ["Parameters", "≈ 1.69 M"],
  ["Result", "test NDCG@5 **0.508** on all users (0.506 / 0.510 / 0.506 by seed); **0.536** on the 100 LLM users"],
  ["Training time", "~15 min per seed on CPU"],
], [2200, 6826]);
P("**Leakage checks we ran** (SASRec, NDCG@5): normal 0.509; given **someone else’s** history 0.181; **no** history 0.130 (≈ random); **shuffled** history 0.434; last item only 0.472. The score collapses without the user’s own history, so the model really uses it and is not reading the answer from somewhere else.");
NOTE("How to say it in the viva.", "“SASRec is the standard Transformer baseline. We trained it with exactly the GRU’s protocol; it scores 0.508, almost the same as the fixed GRU, which independently confirms our trained-model numbers. Our leakage tests show the score drops to random without the user’s own history.”");
H2("6.6 Non-learned baselines");
B([
  "**Random:** shuffled with a seed. Expected Hit@5 = 0.25.",
  "**Popularity:** order by training interaction count.",
  "**Content similarity:** user profile = recency-weighted sum of the MiniLM vectors of their history (weight e^(−0.3·age)); rank candidates by similarity. No training.",
  "**ItemKNN:** item–item cosine similarity from “reviewed by the same users” counts; score = recency-weighted sum of similarities to history items.",
]);
H2("6.7 Late fusion (combining two rankings)");
P("The LLM outputs a ranking, not scores, so we convert each ranking into scores and blend:");
CODE([
  "s(i)     = 1 − (rank(i) − 1) / 19          (top item = 1.0, last = 0.0)",
  "final(i) = α · s_LLM(i) + (1 − α) · s_GRU(i)",
]);
P("α = 1 gives pure ARAG, α = 0 gives pure GRU. α is chosen on validation: **0.2** for ARAG+GRU, **0.1** for ARAG+text-GRU, **0.4** for ARAG+content. Ties are broken by NLI score. A small α means validation trusts the trained model most, but a little ARAG still helps (Figure 5).");

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
  ["`src/arag/models/`", "`gru.py` (model, incl. text features), `train_gru.py`, `sasrec.py` + `train_sasrec.py`, `score.py`"],
  ["`src/arag/ranking/`", "`fusion.py`, `parser.py`, `simple.py` (random/popularity), `content.py` (content, ItemKNN)"],
  ["`src/arag/eval/`", "`run.py` (the whole experiment), `metrics.py`, `bootstrap.py`, `report.py`, `verify.py`, `extra.py` (history length, SASRec, beyond-accuracy)"],
  ["`scripts/`", "`make_figures.py`, `make_guide.js` (this guide), `build_item_extras.py` (product images/ratings for the demo)"],
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
  "`make gru` trains the ID-only GRU (grid + 3 seeds); `make gru-text` trains the text-aware GRU (3 seeds); `make sasrec` trains SASRec (grid + 3 seeds).",
  "`make eval` (`eval/run.py`): tunes θ and α on validation, runs every system on test, fuses, computes metrics and bootstrap CIs, writes `results/metrics.json` and per-user predictions.",
  "`make report` writes `results/tables.md`, `report.md` and the α-curve plot. `make verify` recomputes every metric from raw predictions and checks hashes.",
  "`make extra` (`eval/extra.py`, no LLM calls) adds the history-length, SASRec and beyond-accuracy analyses → `results/extra_tables.md` and Figures 12–14.",
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
  "make gru gru-text sasrec  # ID-only GRU (~2h25 CPU), text-aware GRU (~26 min), SASRec (~50 min)",
  "make eval                 # all systems; LLM calls cached on disk",
  "make report verify        # results/ + full audit (expect: VERIFY OK)",
  "make extra                # history length, SASRec, beyond-accuracy (no LLM calls)",
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
  ["GRU fix, SASRec, extra analyses (October 2026)", "**$0** (replay only; 2,504 cached calls re-used)"],
  ["Demo, per uncached feed page", "7 calls ≈ $0.003 with gpt-4o-mini"],
], [5000, 4026]);

// =====================================================================================
H1("9. Testing the System from the Frontend (Choosing Users)");
P("**The demo is a small online game store (“GameShelf”) driven by the real system.** You sign in as one of **100 real test customers** from the dataset (the same 100 users the LLM systems were evaluated on — no fake users, no made-up products). Product images, star ratings, prices, platforms and descriptions come from the Amazon metadata. The feed is ARAG fused with the **text-aware GRU**; α is read from `results/metrics.json` (`best_alpha_text`, now **0.1**, i.e. mostly the GRU’s ranking, with ARAG’s reasons shown on every card). Set `app.gru_variant: id` in `configs/config.yaml` to use the ID-only GRU instead.");
H2("9.1 Start the demo");
CODE([
  "# Option A — full live demo (new LLM calls allowed, ~$0.003 per new page)",
  "export OPENAI_API_KEY=sk-...",
  "make app",
  "",
  "# Option B — fully offline (only cached LLM answers; uncached requests show an error)",
  "REPLAY_ONLY=1 OPENAI_API_KEY=replay-only-dummy make app",
  "",
  "# then open  http://127.0.0.1:8000     (different port:  PORT=8010 make app)",
]);
P("The first start takes ~20–60 s (frontend build + loading MiniLM, FAISS and the GRU). `make app` also builds the product-display file (`artifacts/item_extras.parquet`) the first time.");
H2("9.2 Step-by-step test script for the team");
N([
  "**Choose a shopper.** The start screen lists the 100 real test customers with how many games each reviewed and their main platform. The account menu (top right) lets you search by ID or platform, sort by history size and switch customers at any time.",
  "**For you (home).** A row of the customer’s real past reviews, then their personalised recommendations. While loading, the page shows the four agents working (UUA → NLI → CSA → IRA). Each card shows the product, rating, price and ARAG’s reason. Click a card for the full product page with features and the evidence the agent used.",
  "**Give feedback:** Like, Save or Dismiss. Dismissed items never appear again for that customer; the GRU’s user vector is recomputed from the updated sequence on the next request (no retraining).",
  "**Search** (top bar): try a natural-language query such as “racing game for PS4 under 30 dollars” or “horror games but not zombies”. The LLM parses keywords, max price and exclusions; FAISS finds matches; ARAG + GRU re-rank them for this customer.",
  "**Taste profile:** view, regenerate or delete the stored UUA profile (what the agent “thinks” about this customer and which events it used).",
  "**Research:** the evaluation results — dataset size, the main table, H1/H2 with confidence intervals and the α curve; token counts instead of dollar cost because replayed runs are free.",
  "**Switch customers** and compare: a PlayStation collector and an accessories buyer get visibly different pages.",
]);
NOTE("Offline caveat.", "In replay mode only requests whose LLM answers are already cached work. Pages and queries tried before replay instantly; new ones need Option A (an API key).", "FFF4E5", "C55A11");
H2("9.3 What to say while demoing");
B([
  "“These are real Amazon customers and real products; the row at the top is what this person actually reviewed.”",
  "“Every card is ranked by four LLM agents plus our trained text-aware GRU, blended with α tuned on validation; the reason under each card is written by the agents.”",
  "“This is exactly where ARAG is valuable even though trained models rank more accurately: explanations and natural-language search.”",
  "“The Research tab shows the same audited numbers as our report.”",
]);

// =====================================================================================
H1("10. Results and How to Read Them");
NOTE("Which numbers are current?", "All numbers below are from the **final run (October 2026)** with the fixed GRU. Older versions of our documents quoted 0.279 / 0.461 for the GRUs and “H1 not supported”; those came from the buggy GRU (Section 12.3).", "FFF4E5", "C55A11");
H2("10.1 All systems on the same 100 test users (NDCG@5)");
TABLE(["System", "Hit@5", "NDCG@5", "Group"], [
  ["Random", "0.220", "0.125", "trivial"],
  ["Popularity", "0.200", "0.109", "trivial"],
  ["LLM-Vanilla-RAG", "0.380", "0.263", "LLM baseline"],
  ["LLM-Recency", "0.400", "0.285", "LLM baseline"],
  ["ARAG w/o NLI & CSA", "0.470", "0.362", "ablation"],
  ["ARAG w/o NLI", "0.510", "0.376", "ablation"],
  ["ItemKNN", "0.540", "0.399", "non-learned"],
  ["ARAG + content (α=0.4)", "0.590", "0.423", "fusion"],
  ["**ARAG**", "0.530", "**0.424**", "agentic LLM"],
  ["Content similarity", "0.590", "0.439", "non-learned"],
  ["Text-aware GRU (3-seed avg)", "0.707", "0.501", "trained (ours)"],
  ["**GRU, ID only (3-seed avg)**", "0.697", "**0.520**", "trained"],
  ["**ARAG + GRU (α=0.2)**", "0.690", "**0.523**", "fusion"],
  ["**ARAG + text-aware GRU (α=0.1)**", "0.720", "**0.532**", "fusion (ours)"],
  ["**SASRec (3-seed avg)**", "0.713", "**0.536**", "trained (Transformer)"],
], [3600, 1500, 1500, 2426]);
P("On these 100 users the five best systems (0.50–0.54) are statistically tied with each other; all of them are above ARAG.");
IMG("fig13_sasrec_baseline.png", "Figure 3 — Every system on the same 100 test users, with 95% confidence intervals.", 6.0);
H2("10.2 Non-LLM systems on all 94,762 test users");
TABLE(["System", "Hit@5", "NDCG@5", "MRR"], [
  ["Random", "0.252", "0.149", "0.180"],
  ["Popularity", "0.213", "0.128", "0.166"],
  ["ItemKNN", "0.519", "0.406", "0.406"],
  ["Content similarity", "0.569", "0.411", "0.402"],
  ["GRU, ID only (3 seeds)", "0.686 ± 0.001", "0.505 ± 0.001", "0.478"],
  ["SASRec (3 seeds)", "0.692", "0.508", "0.479"],
  ["**Text-aware GRU (3 seeds)**", "**0.716 ± 0.001**", "**0.530 ± 0.001**", "**0.500**"],
], [3300, 1900, 1900, 1926]);
IMG("fig10_content_aware.png", "Figure 4 — Left: all 94,762 users. Right: the same 100 users as the LLM systems.", 6.3);
H2("10.3 Significance (paired bootstrap, ΔNDCG@5)");
TABLE(["Contrast", "Δ", "95% CI", "Significant?"], [
  ["H1: ARAG+GRU − ARAG", "+0.098", "[+0.013, +0.182]", "**yes**"],
  ["H2: ARAG+GRU − GRU", "+0.004", "[−0.038, +0.045]", "no"],
  ["ARAG − LLM-Recency", "+0.139", "[+0.063, +0.217]", "**yes**"],
  ["ARAG − LLM-Vanilla-RAG", "+0.162", "[+0.079, +0.247]", "**yes**"],
  ["ARAG − ARAG w/o NLI & CSA", "+0.062", "[+0.016, +0.113]", "**yes**"],
  ["ARAG − ARAG w/o NLI", "+0.049", "[−0.011, +0.111]", "no"],
  ["ARAG − GRU (100 users, 3-seed avg)", "−0.095", "[−0.196, +0.006]", "no (borderline)"],
  ["ARAG − SASRec (100 users)", "−0.112", "[−0.213, −0.009]", "**yes**"],
  ["ARAG+text-GRU − ARAG", "+0.108", "[+0.016, +0.197]", "**yes**"],
  ["ARAG+text-GRU − text-GRU (3-seed avg)", "+0.031", "[+0.002, +0.061]", "**yes** (small)"],
  ["Text-GRU − GRU (all 94,762 users)", "+0.026", "[+0.024, +0.027]", "**yes**"],
  ["Text-GRU − content (all users)", "+0.120", "[+0.117, +0.122]", "**yes**"],
  ["SASRec − GRU (all users)", "+0.003", "[+0.002, +0.004]", "yes, but tiny"],
], [3700, 1300, 2300, 1726]);
IMG("fig03_hypotheses.png", "Figure 5 — Forest plot of the key contrasts. A bar crossing zero means “not significant”.", 5.6);
IMG("fig05_alpha_curve.png", "Figure 6 — ARAG + ID-only GRU vs the fusion weight α (0 = pure GRU, 1 = pure ARAG). Validation picks α = 0.2: mostly GRU, with a little ARAG on top. Pure ARAG (α = 1) is the worst point.", 5.0);
H2("10.4 How to interpret the results in one minute");
N([
  "ARAG is a genuinely strong **LLM** system and the paper’s claims (vs LLM baselines, ablation) reproduce.",
  "Properly trained sequence models (GRU, SASRec, text-GRU) are **more accurate** than ARAG on this task and cost nothing per request.",
  "Fusion still helps ARAG a lot (+0.098) — the two make different mistakes — and adding ARAG to the text-GRU gives a small significant gain (+0.031).",
  "A zero-training content-similarity ranker ties with ARAG: much of ARAG’s skill is “reading item text”.",
  "With only 100 LLM-evaluated users, the top systems cannot be separated from each other; we say so honestly.",
]);
NOTE("How to say it in the viva.", "“ARAG beats every LLM method we tried, but trained sequence models beat ARAG on accuracy. The best accuracy comes from the trained model, and ARAG adds explanations and a small extra gain when fused.”");

H2("10.5 Extra analysis 1 — does history length matter?");
P("We split users by how many items they had before the test item and computed NDCG@5 in each group (zero LLM calls: re-using saved rankings).");
TABLE(["History", "Users", "GRU", "Text-GRU", "SASRec", "Content"], [
  ["4 items", "30,451", "0.497", "0.521", "0.502", "0.420"],
  ["5–6", "29,707", "0.503", "0.530", "0.508", "0.417"],
  ["7–10", "20,678", "0.510", "0.536", "0.511", "0.402"],
  ["11+", "13,926", "0.519", "0.543", "0.513", "0.391"],
], [1700, 1400, 1400, 1500, 1500, 1526]);
B([
  "The trained models are almost **flat**: users with only 4 items get nearly the same quality as users with 11+. So short histories were **not** why our first GRU was weak — that was the initialisation bug.",
  "The text-GRU is ahead of the GRU in **every** group (+0.024 to +0.027, all significant).",
  "Content similarity gets **worse** with longer histories (its averaged profile gets blurry), while ItemKNN gets better (more items to match).",
  "On the 100 LLM users (44 / 30 / 26 users per group) the confidence intervals are wide; ARAG is below the GRU in every group but not significantly.",
]);
IMG("fig12_history_length_performance.png", "Figure 7 — NDCG@5 by history length. Left: all users (tight error bars). Right: the 100 LLM users (wide error bars).", 6.3);
NOTE("How to say it in the viva.", "“We first thought short histories hurt the GRU. We tested that directly: performance is flat across history lengths, so that explanation was wrong — and that is what led us to find the real bug.”");

H2("10.6 Extra analysis 2 — SASRec as an independent check");
P("SASRec (Section 6.5) reaches 0.508 on all users and 0.536 on the 100 LLM users. It is the only system **significantly** better than ARAG on the 100 users (+0.112, CI [+0.009, +0.213]). It is essentially tied with the fixed GRU (+0.003) and below the text-GRU (−0.023). Two different architectures landing at the same score is strong evidence that the trained-model numbers are right.");

H2("10.7 Extra analysis 3 — beyond accuracy");
P("Accuracy is not everything. On the 100 test users we measured three properties of each system’s top 5:");
B([
  "**Popularity lift**: how much more popular the top 5 is than the 20-item list on average (log scale). 0 = no preference for popular items.",
  "**Diversity (ILD)**: 1 − average text similarity between the top-5 items. Higher = more varied.",
  "**Coverage**: how many different items appear in anyone’s top 5, out of all candidate items.",
]);
TABLE(["System", "Popularity lift", "Diversity", "Coverage"], [
  ["Popularity", "1.78", "0.723", "0.181"],
  ["ItemKNN", "0.98", "0.723", "0.241"],
  ["GRU / SASRec / text-GRU", "0.74–0.79", "0.718–0.721", "0.247–0.253"],
  ["ARAG + GRU / + text-GRU", "0.73–0.75", "0.717–0.720", "0.254"],
  ["**ARAG**", "**0.13**", "0.726", "**0.275**"],
  ["Content similarity", "0.08", "0.646", "0.278"],
], [3300, 1900, 1900, 1926]);
B([
  "**ARAG has almost no popularity bias** (0.13 vs ~0.77 for the trained models) and higher coverage: it recommends less-known items more often. Trained models learn “popular things get reviewed”, which is accurate but reinforces popularity.",
  "Diversity is about the same for everyone (~0.72) except content similarity, which picks look-alike items.",
  "Fusion inherits the trained model’s popularity bias at the chosen α.",
]);
IMG("fig14_beyond_accuracy.png", "Figure 8 — Popularity bias (left) and diversity (right) of each system’s top 5.", 6.3);
NOTE("How to say it in the viva.", "“ARAG is less accurate, but it is the least popularity-biased system and covers more of the catalogue, so it is fairer to niche products. That is a real reason to keep an LLM in the loop.”");

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
IMG("fig04_ablation.png", "Figure 9 — Ablation staircase: ours next to the paper’s three categories.", 5.6);
H2("11.1 What the paper did not test");
P("The paper compares ARAG only with **other LLM methods** (Recency, Vanilla RAG and its own ablations). It reports no trained sequential recommender (GRU4Rec, SASRec), no non-LLM baseline and no significance tests. On our data, those missing baselines are exactly the ones that beat ARAG. So our results **do not prove the paper wrong** — its claims are about LLM methods and they hold — but they show the comparison was incomplete.");
NOTE("How to say it in the viva.", "“We reproduce the paper’s claims. Our contribution is to add the baselines it left out; with them, ARAG is the best LLM method but not the most accurate recommender. Its advantages are explanations, natural-language queries and lower popularity bias.”");

// =====================================================================================
H1("12. The Mistakes We Found and Fixed (Errata)");
P("**Be ready to tell this story — professors love it, and it shows scientific honesty.**");
H2("12.1 The first mistake: unfair candidate lists");
P("The first version sampled the 19 negatives **uniformly from all 137,260 catalogue items**. But 81% of those items have **zero** interactions. On average **15.5 of 19 negatives were unknown** to Popularity (all tied at 0) and to the GRU (all mapped to the same “unknown” embedding). Those systems effectively faced a 4.5-way choice instead of a 20-way one. Popularity reached Hit@5 = 0.97, the GRU NDCG@5 = 0.82, and fusion appeared to improve ARAG by +0.349 — a **false “H1 supported”** result.");
H2("12.2 How the lists were fixed");
B([
  "Negatives re-sampled from the 25,527 known items, proportional to popularity (Section 5.2).",
  "Every system re-evaluated; the GRU retrained from scratch; all systems compared on the same 100 users.",
  "The original numbers are kept (not deleted) in REPORT.md Appendix E and the paper’s Appendix B.",
]);
TABLE(["Quantity", "Flawed lists", "Corrected lists"], [
  ["Unknown negatives per list", "15.5 / 19", "0 / 19"],
  ["Popularity NDCG@5", "0.801", "0.109"],
  ["GRU NDCG@5 (all users)", "0.823", "0.279 (buggy GRU) → 0.505 (fixed)"],
  ["H1 (fusion − ARAG)", "+0.349 “significant”", "−0.013 (buggy GRU) → +0.098 significant (fixed)"],
], [3600, 2700, 2726]);
IMG("fig07_protocol_flaw.png", "Figure 10 — The flawed lists vs the corrected lists (right panel shows the final fixed-GRU numbers).", 6.0);
NOTE("Lesson.", "How you build the candidate list can create or erase a whole conclusion. Always check that no system can win through a shortcut the others cannot use (see Krichene & Rendle, KDD 2020, on sampled metrics).");
H2("12.3 The second mistake: the GRU initialisation bug");
P("**How we found it.** We added SASRec as an extra baseline. Its first training run also got stuck (loss ≈ 32, validation ≈ 0.15). The cause was the default N(0, 1) embedding initialisation with a tied output layer (Section 3.6). Our GRU used the same initialisation, so we tested the same fix on it: validation NDCG@5 went from **0.293 to 0.513** in a short test run. Our “short histories” explanation was also contradicted by the history-length analysis (Section 10.5): the old GRU was flat across history lengths too.");
P("**What we did.** One-line fix in `src/arag/models/gru.py` (small N(0, 0.02) init, padding row zero), retrained the ID-only GRU (same grid, 3 seeds) and the text-aware GRU, raised the epoch cap from 12 to 30 for both (early stopping still decides), re-tuned α on validation and re-ran the whole evaluation in replay mode (**$0**, every LLM answer from the cache). `make verify` passes. The old models and numbers are kept in `artifacts/backup_v1/`.");
TABLE(["Quantity", "Buggy GRU", "Fixed GRU"], [
  ["GRU validation NDCG@5 (seed 0)", "0.293", "0.528"],
  ["GRU test NDCG@5, all users", "0.279", "0.505"],
  ["Text-GRU test NDCG@5, all users", "0.461", "0.530"],
  ["GRU on the 100 users", "0.238", "0.520"],
  ["α* (ARAG + GRU)", "0.9 (mostly ARAG)", "0.2 (mostly GRU)"],
  ["H1: ARAG+GRU − ARAG", "−0.013, n.s.", "**+0.098, significant**"],
  ["H2: ARAG+GRU − GRU", "+0.174, significant", "+0.004, n.s."],
], [3600, 2700, 2726]);
NOTE("How to say it in the viva.", "“Our first GRU had a weight-initialisation bug that made it look weak. We found it while adding SASRec, fixed it, retrained, and re-ran everything at zero cost. Both hypotheses flipped: fusion now helps ARAG, but does not beat the GRU alone. We kept the old numbers and documented the change.”");
P("We also fixed a smaller statistics issue at the same time: for the 94,762-user comparisons the bootstrap used a random 20,000-user subset, which made those confidence intervals about twice too wide. It now uses all users exactly (Section 3.10). No 100-user number changed.");

// =====================================================================================
H1("13. Limitations and Future Work");
TABLE(["Limitation", "Why it matters / what we would do"], [
  ["Only 100 LLM-evaluated users", "CIs ≈ ±0.08–0.10; top systems cannot be separated. ~300 users (~$3) would resolve ~0.05 differences."],
  ["Sampled negatives", "Easier than real retrieval candidates; results measure ranking on hard 20-item lists, not end-to-end retrieval."],
  ["Negatives slightly more popular than targets", "Popularity scores slightly below random; slightly disadvantages frequency-sensitive models (incl. GRUs). Effect ~0.04 Hit@5, smaller than our key gaps."],
  ["Hyper-parameter search is small", "GRU grid of 4, SASRec grid of 2, text-GRU fixed config. Larger searches could move the trained models a little; they would not help ARAG."],
  ["Leave-last-out leakage", "Other users’ later interactions can be in GRU/ItemKNN training; inherent to the official split, equal across those systems."],
  ["ID-only GRU still improving at 30 epochs", "Its validation score was still creeping up (0.528 → 0.529); if anything the trained models are slightly under-estimated."],
  ["One category, one LLM", "Results may not transfer; future: more categories, other LLMs."],
  ["Offline metrics only", "No user study or A/B test."],
], [3200, 5826]);
P("**Future work:** SASRec with the same text-aware item vectors; review text as an extra item feature; feed the GRU’s scores to the IRA agent as a feature instead of fusing afterwards; retriever-mined negatives; more LLM-evaluated users.");

// =====================================================================================
H1("14. Professor Q&A Bank");
P("Answers are written to be said out loud. Numbers are exact; memorise the ones in bold.");

H2("14.1 Motivation and scope");
QA("What problem does your project solve?", "Next-item recommendation: given a user’s Amazon review history (Video Games), rank 20 candidate products so the item they actually reviewed next is near the top. We compare an agentic LLM recommender (ARAG) with a trained deep sequence model (GRU), and their combination.");
QA("Why is this a deep-learning project if you use GPT?", "The trained component is ours: a PyTorch GRU (≈1.7 M parameters) trained with a full-softmax next-item objective, plus our text-aware extension that adds a learned projection of frozen transformer (MiniLM) embeddings. We also trained a Transformer (SASRec) under the same protocol. The LLM is a strong baseline and fusion partner; the most accurate systems (0.505–0.530 NDCG@5 on all users) are the trained models.");
QA("What was your hypothesis?", ["H1: fusing the GRU into ARAG beats ARAG alone. H2: the fused system beats the GRU alone. **H1 holds (+0.098, significant); H2 does not (+0.004, n.s.).**", "Be ready to say that this flipped: with our first, buggy GRU it was the other way round (Section 12.3)."]);
QA("Is this a reproduction of the ARAG paper?", "It is ARAG-inspired, not an exact reproduction: the paper did not publish prompts, its threshold or candidate construction; it used gpt-3.5-turbo on 2016 data. We test whether its claims hold, and they do: ARAG beats Recency and Vanilla-RAG significantly and the NLI+CSA ablation hurts significantly.");
QA("What is new in your work compared to the paper?", ["(1) Fusion with a trained GRU. (2) Statistical testing (paired bootstrap) on every contrast. (3) Non-LLM and trained baselines the paper left out: popularity, content, ItemKNN, GRU, SASRec. (4) The text-aware GRU, the most accurate single model, at zero API cost. (5) History-length and beyond-accuracy analyses. (6) Two documented mistakes and their fixes."]);

H2("14.2 Data");
QA("Which dataset and why?", "Amazon Reviews 2023, Video Games, official 5-core leave-last-out split: real data, an official split (no custom leakage), rich item metadata for the LLM, and a category the ARAG paper never ran.");
QA("How big is it?", "**94,762 users, 814,586 interactions, 25,527 items in the training vocabulary.** Median training history: **4 items**.");
QA("What does 5-core mean?", "Every user and item has at least 5 interactions. After holding out the validation and test items, a user can have as few as 3 training items.");
QA("Do you use review text?", "No — only product metadata (title, categories, features, description, ≤120 words). Using review text could leak the target item.");
QA("Your demo user has only 4–5 items — is that accurate?", ["Yes. The data records **reviews**, not purchases; a user may buy far more but review few. It is the official 5-core benchmark, so every user has at least 5 reviews (one is held out for test, one for validation), and the median user has 5. Histories go up to 472 items; our models use up to the last 50.", "That sparsity is realistic. Our history-length analysis shows the trained models work almost equally well for 4-item and 11+-item users."]);
QA("How do you prevent data leakage in the split?", "We use the official leave-last-out files and assert max(train time) ≤ validation time ≤ test time, and that targets never appear in their own history; the pipeline fails hard otherwise. One known caveat: other users’ later interactions can appear in training (inherent to this split), affecting the trained models equally.");

H2("14.3 Evaluation protocol");
QA("Why do all systems rank the same 20 candidates?", "To measure ranking, not retrieval. If each system retrieved its own candidates, a retrieval miss would count as a ranking failure.");
QA("How are the 19 negatives chosen and why?", "From the 25,527 items seen in training, proportional to popularity, excluding the user’s own items. This makes decoys about as popular as targets, so popularity alone gives no advantage — popularity scores 0.109, below random (0.125).");
QA("Why only 100 users for the LLM systems?", "Cost and time: ARAG makes 7 calls per user. 100 users keeps each run around $1. All systems are compared on the same 100 users, and non-LLM systems are additionally evaluated on all 94,762 users.");
QA("Isn’t 100 users too few?", "It is enough to detect large effects (ARAG vs the LLM baselines, ARAG vs SASRec, fusion vs ARAG, the ablation) but not small ones — the CI is about ±0.08–0.10. That is why we report CIs and say the top systems are statistically tied. ~300 users would resolve ~0.05.");
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
QA("Describe your GRU architecture.", "Item embedding (64, initialised N(0, 0.02)) → dropout → 1-layer GRU (hidden 64, chosen by grid) → dropout → linear projection to 64; the score is the dot product with tied item embeddings; full-softmax cross-entropy over 25,527 items at every position; Adam, lr 1e-3, batch 256, ≤30 epochs with early stopping on validation NDCG@5.");
QA("Why a GRU and not an LSTM or Transformer?", "GRU4Rec is the classic deep sequential recommender, with fewer parameters than an LSTM and easy to train on CPU. We also trained the Transformer alternative, SASRec, under the same protocol: 0.508 vs the GRU’s 0.505 — essentially the same, because with median history 4 there is little long-range structure for attention to exploit.");
QA("Your first GRU scored 0.279. What went wrong?", ["A weight-initialisation bug. PyTorch’s default N(0, 1) embedding init with tied input/output embeddings makes the initial logits huge, so the softmax saturates and training stalls. We found it when SASRec showed the same symptom. With N(0, 0.02) init the GRU reaches 0.505.", "We originally blamed short histories; our history-length analysis disproved that, because performance is flat across history lengths."]);
QA("What exactly is the text-aware GRU?", "Same GRU, but each item vector is e(i) + W·MiniLM(text_i): a learned ID embedding plus a learned 384→64 projection of the item’s frozen MiniLM text embedding, used for both input and tied output. Only 24,576 extra parameters.");
QA("Why does adding text help so much?", "It lets the model generalise across items: two products with similar descriptions get similar vectors even if they rarely co-occur. With 3–5 interactions per user, this shared content signal fills in what the ID embeddings cannot learn.");
QA("Is the comparison between the two GRUs fair?", "Mostly: same code, optimiser, initialisation, epoch budget, early-stopping rule and seeds. One difference: the text-GRU uses hidden size 128 (the best setting from the first grid) while the re-run grid picked 64 for the ID-only GRU. Both settings were within 0.02 on validation.");
QA("Did the text-aware GRU overfit?", "Early stopping on validation stopped it after ~13 epochs at 0.547–0.550; test is 0.530 ± 0.001 across 3 seeds. The small validation–test gap is expected (the test target is one step later in time). No sign of harmful overfitting.");
QA("Why freeze MiniLM?", "Cheaper and safer: the text encoder is pre-trained on general language; fine-tuning it on 625k short sequences on a CPU risks overfitting and costs much more. The learned projection W adapts it to our task.");
QA("How many parameters?", "ID-only GRU ≈ **1.66 M** (1.63 M in the item embedding table). Text-aware GRU ≈ **1.74 M** (hidden 128, +24,576 for W). SASRec ≈ **1.69 M**. The item table dominates all three.");
QA("How long does training take?", "On a laptop CPU: ID-only grid + 3 seeds ≈ 2 h 25 min (up to 30 epochs); text-aware GRU ≈ 8–9 min per seed; SASRec ≈ 15 min per seed.");
QA("What memory problems did you hit?", "Training was killed for running out of memory on 16 GB. Two fixes: compute logits only at real (non-padded) positions, and encode validation users in chunks of 4,096 with history capped at 50. Same loss; the original GRU’s validation score was reproduced exactly (0.2928) as a check.");

H2("14.6 Fusion");
QA("How do you combine the LLM and the GRU?", "Late fusion of rankings: convert each rank to a score 1 − (rank−1)/19 and blend final = α·LLM + (1−α)·GRU. α = 1 is pure ARAG, α = 0 is pure GRU; α is tuned on validation.");
QA("Why late fusion instead of feeding the GRU into the LLM?", "The LLM outputs a permutation, not scores, and late fusion makes the GRU’s contribution exactly ablatable (α = 1 recovers ARAG). Giving GRU scores to the ranker agent as a feature is listed as future work.");
QA("Does ARAG + GRU help?", "It helps **ARAG**: +0.098 NDCG@5, significant. It does not beat the GRU alone (+0.004, n.s.). Validation chose α = 0.2, i.e. 80% GRU. The two systems make different mistakes, so a little of ARAG’s ranking on top of the GRU is safe, but the GRU carries the accuracy.");
QA("Did fusion help with the text-aware GRU?", "Yes, slightly: ARAG + text-GRU (α = 0.1) is 0.532 vs 0.501 for the text-GRU alone on the 100 users (+0.031, CI [+0.002, +0.061]) and +0.108 over ARAG. It is statistically tied with SASRec (0.536).");
QA("Why did α change from 0.9 to 0.2?", "α is chosen on validation and reflects how good each signal is. With the buggy GRU, validation trusted the LLM (0.9). With the fixed GRU, validation trusts the GRU (0.2). This is the fusion working as designed: it follows the stronger model.");

H2("14.7 Results and interpretation");
QA("What is your best system?", "On all 94,762 users: the text-aware GRU (0.530 NDCG@5). On the 100-user sample: SASRec (0.536) and ARAG + text-GRU (0.532), statistically tied with the GRU (0.520) and the text-GRU (0.501). All of them beat ARAG (0.424).");
QA("So is the LLM useless?", ["No. Among LLM methods the agentic design clearly matters (ARAG beats Recency by +0.139 and Vanilla-RAG by +0.162, significant). ARAG explains every recommendation, handles natural-language search, has almost no popularity bias (0.13 vs ~0.77) and adds a significant gain when fused into the GRU.", "But for pure ranking accuracy on this data, a trained model is better and costs nothing per request."]);
QA("Does this prove the ARAG paper wrong?", "No. The paper claims ARAG beats other LLM recommenders, and that reproduces. It did not compare against trained sequence models. We show those are more accurate here — one category, one LLM, 100 users — so the honest statement is “the paper’s comparison was incomplete”, not “the paper is wrong”.");
QA("Why does popularity score below random?", "Negatives are sampled proportional to popularity and end up on average slightly more popular than targets (1.01× validation, 1.40× test). Ranking by popularity therefore pushes decoys up slightly. It confirms the lists remove the popularity shortcut.");
QA("How confident are you in the main result?", "Very, for the trained models: 3 seeds agree to ±0.001, two architectures (GRU, SASRec) agree, and CIs on 94,762 users are ±0.002. For ARAG comparisons n = 100 gives wide CIs: ARAG vs SASRec is significant, ARAG vs GRU is borderline (CI just touches 0). We state this explicitly.");
QA("What is the cost–quality trade-off?", "ARAG: 7 API calls per user, seconds of latency, ≈ $0.003 per page. Trained models: score all 94,762 users in seconds on a CPU, train in minutes to hours, $0 per request — and more accurate here.");

H2("14.8 Engineering, reproducibility and ethics");
QA("How can we verify your numbers?", "`make verify` recomputes every metric from the saved per-user rankings, checks dataset and list hashes against the run record, and prints VERIFY OK. All tables in the paper are generated from `results/metrics.json` by a script.");
QA("How much did it cost?", "**$2.04** in total LLM spend (4,668 calls). All extension experiments — text-GRU, the GRU fix, SASRec, the extra analyses — cost $0 thanks to the cache.");
QA("Tell us about a mistake you made.", ["Two. (1) Our first candidate lists drew negatives from the whole catalogue, 81% of which had no interactions, so popularity and the GRU could spot the target easily and we wrongly concluded H1 was supported (+0.349). We rebuilt the lists and re-ran everything.", "(2) Our GRU had a weight-initialisation bug that made it look weak (0.279); we blamed short histories. Adding SASRec exposed the bug; the fixed GRU scores 0.505 and the hypotheses flipped. Both are documented with the old numbers kept."]);
QA("What tests do you have?", "A pytest suite: metrics, bootstrap, fusion (α = 0/1 reproduce each input exactly), the parser, the lists, GRU padding invariance and save/load, the text-feature model, content/ItemKNN, the API and the agents (the last two use a local Ollama model).");
QA("Any privacy or ethical concerns?", "The data is public and pseudonymised (hashed user IDs). LLM prompts contain only product metadata and anonymised histories. Recommenders can reinforce popularity bias; our protocol explicitly removes the popularity shortcut in evaluation.");

H2("14.9 Curve-ball questions");
QA("If you had one more week, what would you do?", "Evaluate 300+ users with the LLM to separate the top systems, give SASRec the same text-aware item vectors, add review text as an item feature, and try giving the GRU’s scores to the ranker agent directly.");
QA("Would the results hold with a better LLM (e.g. GPT-4o)?", "Possibly ARAG would improve; but the content-similarity result suggests much of the gain comes from reading item text, which a trained model can also do. It is an open question we would test.");
QA("Why not just use content similarity, since it ties with ARAG?", "Content similarity ignores collaborative patterns. The text-aware GRU combines both and significantly beats content similarity on all users (+0.120, CI [+0.117, +0.122]).");
QA("Is NDCG@5 on 20 sampled candidates realistic?", "It is a controlled ranking benchmark. Real systems rank hundreds of retrieved items; sampled metrics can distort rankings (Krichene & Rendle, 2020). We acknowledge this as a limitation and designed the negatives to be hard.");
QA("Could the text-aware GRU be cheating by seeing the target’s text?", "No. It sees the text of every vocabulary item, exactly as ARAG sees every candidate’s text; it never sees which candidate is the target, and no review text or held-out interactions are used.");

H2("14.10 The new analyses (SASRec, history length, beyond accuracy)");
QA("What is SASRec and why did you add it?", "Self-Attentive Sequential Recommendation (Kang & McAuley, 2018), the standard Transformer recommender. We added it as an independent strong baseline under exactly the GRU’s protocol. It scores 0.508 on all users and 0.536 on the 100 LLM users, significantly above ARAG (+0.112).");
QA("How do you know SASRec is not leaking the answer?", "We ran ablations at test time: with someone else’s history it drops from 0.509 to 0.181, with no history to 0.130 (≈ random), with a shuffled history to 0.434. Only 0.4% of targets and 0% of negatives are out-of-vocabulary. The score depends on the user’s own history, as it should.");
QA("Does performance depend on history length?", "Very little for the trained models: GRU 0.497 for 4-item users vs 0.519 for 11+; text-GRU 0.521 vs 0.543. Content similarity actually gets worse for long histories. So short histories are not a problem for a properly trained sequence model.");
QA("What is popularity lift and why does ARAG score so low?", "The average log-popularity of the top 5 minus that of the 20-item list. Trained models (~0.77) and ItemKNN (0.98) prefer popular items because they learn from interaction counts. ARAG (0.13) judges items by their descriptions against the user’s profile, so it has almost no popularity bias and covers more of the catalogue (0.275 vs ~0.25).");
QA("Is ARAG more diverse?", "Barely: intra-list diversity is ~0.72 for almost every system (ARAG 0.726). Only content similarity is clearly less diverse (0.646) because it picks look-alike items.");
QA("Why didn’t you use review text for the GRU?", "We use product metadata only (title, categories, features, description), the same information ARAG sees. Review text is a possible future extension; it must be restricted to reviews written before the test time to avoid leakage.");
QA("You re-ran everything — did you spend more on the LLM?", "No. Every LLM answer is cached on disk keyed by the hash of the prompt. The re-evaluation ran with `REPLAY_ONLY=1`, which makes any uncached call an error. Cost: $0.00.");

// =====================================================================================
H1("15. Glossary");
TABLE(["Term", "Meaning"], [
  ["ARAG", "Agentic Retrieval-Augmented Generation recommender: UUA, NLI, CSA, IRA agents"],
  ["UUA / NLI / CSA / IRA", "User Understanding / Natural Language Inference / Context Summary / Item Ranker agents"],
  ["GRU", "Gated Recurrent Unit, a recurrent neural network for sequences"],
  ["GRU4Rec", "Hidasi et al. 2016: GRU for session-based recommendation"],
  ["Text-aware GRU", "our GRU whose item vectors = ID embedding + projected MiniLM text embedding"],
  ["SASRec", "Kang & McAuley 2018: Transformer (self-attention) model for sequential recommendation"],
  ["Self-attention", "each position computes a weighted average of earlier positions, with learned weights"],
  ["Initialisation", "the random starting values of the weights; N(0, 0.02) = small normal values"],
  ["Popularity lift", "how much a top-5 list prefers popular items (0 = no preference)"],
  ["ILD (intra-list diversity)", "1 − average similarity between items in a list"],
  ["Coverage", "fraction of candidate items that appear in anyone’s top 5"],
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
  ["Data & protocol", "dataset stats, frozen lists, negatives, metrics, bootstrap, both errata stories", "4, 5, 3.9–3.10, 12"],
  ["LLM side", "ARAG agents, θ, prompts, baselines, ablations, paper comparison, beyond-accuracy, cost", "6.1–6.2, 10.7, 11, 8.5"],
  ["Deep-learning side", "GRU maths, initialisation bug, training, text-aware GRU, SASRec, results", "3.4–3.6, 6.3–6.5, 10, 12.3"],
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
