# DADA2 — Agentic RAG over SEC filings

**Fill in both sections at the bottom before you submit.** Keep the deployed
app line exactly in this form; it is how your app gets found and graded.

## Getting started

There is no separate spec. The instructions are in the notebooks, and
`DADA2_00_Start_Here.ipynb` is the one to open first: it explains the
assignment, the setup, the grading, and what each of the four notebooks after
it asks for. Work through them in order.

Get the files the same way as every class day: pull, then work in a copy:

```bash
cd lecture-files
git pull
cd ..
cp -r lecture-files/DADA2 DADA2     # work in the copy, not in lecture-files/
cd DADA2
cp example.env .env                  # then fill in your keys
```

Then open `DADA2_00_Start_Here.ipynb` from your `DADA2` copy and run its
setup check.

Deploying is `DEPLOY.md`.

---

## Submission

```
Deployed app: https://sec-filings-agent-muhammet.onrender.com
```

ADD YOUR NAME TO YOUR RENDER PROJECT SO THE LINK CAN BE PERSONALIZED.

PLEASE ENSURE YOUR URL WORKS BY TESTING IT IN INCOGNITO MODE.

### WHAT TO SUBMIT
Upload the entire project folder (DADA2) via GitHub integration on Canvas

### Write-up

Paste the write-up from notebook 04 (TODO 4.3) here, under the same five
headings.

**Conditions.**

Cohere rerank-v3.5, chat model gemini-3.5-flash-lite, decision model clef-flash, Cloudflare qwen3-embedding-0.6b. Chunks are 1,200 characters with 150 overlap (2,124 chunks), K = 5, pool = 10, RRF constant 60; router falls back to filter+rrf below 0.6; routers tested on 24 questions and agents were tested on 14 (8 were specific 6 were comparative). Answers were judged by Gemini. In a re-run of 4 agent questions, the judge marked a diagnosis-agent answer as WRONG for citing "fiscal 2025", but the corpus includes AMD's 2025 filings. Since that was one of the 14 questions the agent scores could have the same judging error.

**The routers.**

The decision model got 20/24 right at 0.31s a call, gemini with structured output got 19/24 at 0.69 sec, so decision model is one question more accurate and around twice the speed. Real value is the probability it returns, allowing the router to be at a safe default when faced with uncertainty.

**The agents.**

Both scored 0.86 (12/14); the ladder averaged 4.1 chat calls and 16.8s per question; diagnosis agent averaged 2.3 chat calls, 2.1 decision calls and 8.8s. Notebook 03 (cells 27 and 29) shows where the savings come from, using a question outside the 14-question set: "AMD's employee dress code," which the filings don't answer. The ladder climbed all 3 rungs and spent 9 chat calls, diagnose returned absent with .92 probability which is above the agent's ABSENT_SURE cutoff of 0.8 - meaning agent answered practically instantly with 1 chat call.

**What did not help.**

RRF and reranking on top of the filter. On the 6-question sample, MRR went from 0.917 (filter, cell 27) to 0.833 (+RRF, cell 27) to 0.806 (+rerank, cell 34), and recall@5 stayed at 1.00. The drops match up with a correct passage moving down a rank on one question. Once the filter narrows the search to the right company and year, a passage from the right file is already near the top. Fusion mixes rewritten queries that rank it lower, and the reranker can only reorder those 10 passages. Without the filter, reranking helped: baseline MRR rose from 0.325 to 0.583 (cell 37). I would remove reranking from the specific-question path only because it only hurts when retrieval is already precise. Filter+rrf+rerank stays in the agent's STRONGER path. RRF stays because it's the DEFAULT and the first STRONGER step.

**What I would ship.**

The diagnosis agent since it gets identical accuracy with 44% fewer chat calls and half the latency. And START_WITH already sends specific questions to filter alone and comparative ones to hybrid+rrf. The hybrid+rrf choice is a design decision, not a measured result, because I only scored baseline and filter on the comparative set (cell 49). I would want to know how many of the user questions are comparative or unanswerable and whether users name the company and year before shipping, however.
