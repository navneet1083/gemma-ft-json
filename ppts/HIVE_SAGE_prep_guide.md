# HIVE & SAGE — Deep-Dive & Interview Prep Guide

*Prepared for Navneet Singh — for frontier-lab applications (DeepMind / Meta FAIR / Anthropic)*

---

## 0. How to use this (read first)

HIVE and SAGE are listed on your résumé as **"In development."** This guide is built to make that true and defensible. It does three things:

1. **Explains each system deeply** so you can whiteboard it and defend every design choice.
2. **Gives you a minimal prototype spec** so "in development" becomes literal — and so you can add real numbers later.
3. **Grounds every claim in real, citable literature** (the "artifacts" you asked for), and points you at hands-on courses/videos to skill up fast.

**One rule for interviews:** talk about HIVE and SAGE as *designs you are building and reasoning about*, not as shipped systems with measured wins. The strength is research taste and systems judgment — that is exactly what wins the conversation when you're up against PhDs. Never quote a metric you haven't measured.

---

## 1. Naming conventions — why these names, and how to talk about them

Strong project names at research labs follow one of two patterns: **(a) an evocative one-word metaphor** (Voyager, Reflexion, Toolformer) or **(b) a crisp acronym that states the function** (ReAct, RAG, DPO). HIVE and SAGE deliberately do both — a memorable metaphor *and* a backronym that signals the mechanism.

| Name | Metaphor | Backronym (interview-ready) | Why it lands |
|---|---|---|---|
| **HIVE** | A bee colony — many simple agents, emergent collective intelligence | **H**eterogeneous **I**ntelligent agents, **V**erified **E**xecution | Swarm = decentralized + emergent; "verified" signals you took safety seriously |
| **SAGE** | A sage who has read the literature and advises wisely | **S**elf-improving **A**gentic **G**rounded **E**ngine | "Grounded" = arXiv-grounded decisions; "self-improving" = the optimization loop |

**If asked "why this name?"** — answer in one breath: *"HIVE is a decentralized agent swarm, so I wanted a name that signals emergence from many simple units rather than a hardcoded org-chart; the backronym also flags that every agent interaction is contract-verified, which is the part most multi-agent systems get wrong."* That answer shows you know the failure modes, not just the buzzword.

**Backup names** (in case an interviewer or another team already uses HIVE/SAGE internally):

- HIVE alternatives: **CHORUS** (coordinated voices), **AGORA** (a marketplace where agents bid for work), **MURMUR** (after starling murmurations — emergent flocking).
- SAGE alternatives: **HELIX** (the iterative refinement loop), **ATLAS**, **COMPASS** (navigating a model/adapter search space).

---

## 2. HIVE — Decentralized Agent Swarm for Complex Task Decomposition

### 2.1 The elevator pitch (memorize this)

> *"Most multi-agent frameworks are statically wired — the developer hardcodes which agent calls which. HIVE is decentralized: lightweight specialist agents register their capabilities, bid for sub-tasks through a market mechanism, coordinate through a shared blackboard, and reach emergent consensus — with no fixed topology. When an agent fails, the work is re-auctioned and reassigned, so the swarm self-heals. Every agent-to-agent handoff passes a formally verified contract, which kills the most common multi-agent failure: malformed data silently cascading downstream."*

### 2.2 The problem it solves

Static multi-agent systems (AutoGen GroupChat, CrewAI crews, LangGraph graphs) work for predictable workflows but break on novel tasks because the orchestration topology is fixed by a human. They also fail silently when one agent passes badly-shaped data to the next. HIVE attacks both: **dynamic composition** (no hardcoded wiring) and **contract verification** (no silent data corruption).

### 2.3 Architecture — the five things to be able to draw

1. **Capability registry.** Each agent publishes a contract: capabilities, input/output JSON schemas, pre/post-conditions, a cost-per-call estimate, and a trust score updated from past performance.
2. **Market-based task allocation (Contract-Net).** The task is decomposed into sub-tasks; agents *bid* (capability match × trust × cost); the best bid wins. This is the classic **Contract Net Protocol** (Reid G. Smith, 1980) applied to LLM agents.
3. **Stigmergic coordination via a blackboard.** Agents read/write to a shared blackboard / vector memory rather than messaging point-to-point — coordination emerges from the shared state, like ants leaving pheromone trails (*stigmergy*, Grassé 1959).
4. **Contract verification.** Before any handoff, a verifier checks the sender's output schema satisfies the receiver's input schema and that postconditions entail preconditions. Catches malformed-data cascades up front.
5. **Safety + self-healing.** Budget/step guards, action allowlists, sandboxed tool execution; on failure, the sub-task is re-auctioned to the next-best agent.

### 2.4 Key design decisions & the tradeoffs (interviewers probe these)

- **Decentralized vs. orchestrator-led.** Decentralized = robust + adaptive, but harder to debug and reason about globally. *Your answer:* HIVE keeps a thin control plane (registry + verifier + monitor) for observability while the data plane stays decentralized — best of both.
- **Market/auction vs. learned router.** Auctions are interpretable and need no training data; a learned router can be better but needs labels. *Your answer:* start with contract-net for cold-start interpretability, then optionally distill a learned bidder once you have logs.
- **Blackboard vs. direct messaging.** Blackboard scales coordination and decouples agents but can become a contention bottleneck. *Your answer:* shard the blackboard by task namespace; use vector retrieval so agents pull only relevant state.
- **Cost.** Every agent turn is an LLM call — a 4-agent, 5-round interaction is ≥20 calls. *Your answer:* use small models for most agents, reserve a strong model for the planner/critic, and cap with budget guards (this is the real production lesson — see the framework comparisons in §5).

### 2.5 The hard problems (and your crisp answers)

- **Emergent loops / non-termination** → step + budget guards, and a monitor agent that detects oscillation.
- **Trust & security (a compromised agent)** → trust scores, contract verification, and action allowlists; cite the *attention-based trust management* and *agentic-network security* work in §5.
- **Evaluation** → don't just measure task-completion; measure **step efficiency** (actual/optimal steps), **tool-call accuracy**, and **recovery rate**. This framing alone signals seniority.

### 2.6 Grounded in (artifacts — cite these)

- **Contract Net Protocol** — Reid G. Smith (1980), *IEEE Transactions on Computers*. The foundation for market-based task allocation. (Search: "Contract Net Protocol Smith 1980".)
- **Stigmergy / Ant Colony Optimization** — Dorigo et al. The basis for blackboard / pheromone-style coordination.
- [Multi-Agent Systems Powered by LLMs: Applications in Swarm Intelligence (arXiv:2503.03800)](https://arxiv.org/abs/2503.03800)
- [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence (arXiv:2410.11163)](https://arxiv.org/abs/2410.11163)
- [LLM2Swarm: robot swarms that reason, plan, and collaborate through LLMs (arXiv:2410.11387)](https://arxiv.org/abs/2410.11387)
- [Exploring advanced LLM multi-agent systems based on blackboard architecture (arXiv:2507.01701)](https://arxiv.org/abs/2507.01701)
- [A Survey on LLM-based Multi-Agent Systems (arXiv:2412.17481)](https://arxiv.org/abs/2412.17481) · [Multi-Agent Collaboration Mechanisms: A Survey (arXiv:2501.06322)](https://arxiv.org/abs/2501.06322) · [LLM Multi-Agent Systems: Challenges and Open Problems (arXiv:2402.03578)](https://arxiv.org/abs/2402.03578)
- Pattern lineage: [ReAct (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629), [Reflexion (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366), [AutoGen (arXiv:2308.08155)](https://arxiv.org/abs/2308.08155), [MetaGPT (arXiv:2308.00352)](https://arxiv.org/abs/2308.00352), [CAMEL (arXiv:2303.17760)](https://arxiv.org/abs/2303.17760), [Generative Agents (arXiv:2304.03442)](https://arxiv.org/abs/2304.03442).
- Frameworks to name-drop / interoperate with: OpenAI **Swarm** (github.com/openai/swarm), Google **A2A** (Agent-to-Agent) protocol, **LangGraph**, **CrewAI**, **AutoGen/AG2**.

### 2.7 Minimal prototype to build (a weekend → makes "in development" real)

Build this and you can speak from experience, not theory:

- **3–4 toy agents** (e.g., *researcher, analyst, writer, critic*) over LangGraph or the OpenAI Agents SDK.
- **A blackboard** = a dict or a tiny vector store (FAISS/Chroma) all agents read/write.
- **A contract-net loop**: decompose a task → broadcast sub-tasks → each agent returns a bid (a number) → assign to highest bid → execute → write result to blackboard.
- **A JSON-schema verifier** on each handoff (use `jsonschema`).
- **A monitor** that enforces a step budget and re-auctions on failure.
- **Eval**: run 20 tasks; log task-completion, step-efficiency, and recovery rate. *Now you have real numbers.*

---

## 3. SAGE — Literature-Grounded Agentic AutoML for Adaptation

### 3.1 The elevator pitch (memorize this)

> *"Picking the right small model and the right adaptation strategy for a new task is usually expert guesswork. SAGE automates it. You give it a task spec and a few sample records; a swarm of agents reads the relevant arXiv literature, profiles candidate models from an internal pool, runs short proxy fine-tunes, and converges on a Pareto-optimal choice across quality, parameter budget, and latency — including whether to use full fine-tuning, LoRA, QLoRA, or projection layers, and at what rank. Every recommendation is traceable to the papers and the proxy runs that justified it."*

### 3.2 The problem it solves

Two real, expensive decisions are made by intuition today:
1. **Which model?** (the *routing/selection* problem)
2. **How to adapt it?** (full FT vs. LoRA vs. QLoRA vs. projection/adapter — and which rank per layer)

SAGE turns both into a grounded, automated search instead of tribal knowledge.

### 3.3 Architecture — the five things to be able to draw

1. **Knowledge base.** Model registry (cards, eval profiles, checkpoints) + a curated **arXiv corpus**, both embedded in a vector DB.
2. **Planner agent.** Reads the task + retrieves relevant methods/SOTA baselines from the corpus.
3. **Selector agent.** Proposes candidate (model, adaptation-strategy, rank) configs — grounded in routing literature (RouteLLM-style cost/quality reasoning) and PEFT literature (AdaLoRA/DyLoRA-style rank allocation).
4. **Trainer + Evaluator agents.** Run **short proxy fine-tunes** on the sample data; measure quality, params, latency.
5. **Self-improving loop.** Iterates toward a **Pareto-optimal** config; logs every decision with its supporting evidence. Reuses your **TARS** per-layer rank-selection idea.

### 3.4 Key design decisions & tradeoffs

- **Proxy fine-tunes vs. full runs.** Proxy (small data / few steps) is cheap but noisy. *Your answer:* use proxy for ranking candidates, confirm the winner with a fuller run — a successive-halving / Hyperband-style budget.
- **LoRA vs. QLoRA vs. full FT decision.** Driven by VRAM budget, data size, and quality target. *Your answer:* SAGE encodes this as a constrained optimization, not a coin-flip — and grounds rank choice in adaptive-rank methods rather than a fixed r=8.
- **Search cost.** AutoML can be expensive. *Your answer:* bandit-style allocation (spend compute on promising configs), and warm-start from the arXiv-retrieved priors so you don't search blind.
- **Why agents at all (vs. a script)?** Because the *grounding* step — reading and synthesizing relevant papers per task — is genuinely an LLM job, and the search benefits from reflection between rounds.

### 3.5 The hard problems (and your answers)

- **Noisy proxy signals** → multi-fidelity evaluation (Hyperband); confirm before committing.
- **Reproducibility** → seed control, logged configs, content-addressable run versions.
- **Avoiding "paper hallucination"** → retrieval-grounded citations only; the evaluator checks claims against actual proxy results, not just what a paper asserts.

### 3.6 Grounded in (artifacts — cite these)

**Model selection / routing**
- [RouteLLM: Learning to Route LLMs with Preference Data (arXiv:2406.18665)](https://arxiv.org/abs/2406.18665) · [LMSYS RouteLLM blog](https://www.lmsys.org/blog/2024-07-01-routellm/)
- [BEST-Route: model + #responses by query difficulty (arXiv:2506.22716)](https://arxiv.org/abs/2506.22716) (code: github.com/microsoft/best-route-llm)
- FrugalGPT (LLM cascades), GraphRouter, Hybrid LLM — the cost/quality routing family.

**Adaptation / PEFT rank decisions (this is where TARS lives)**
- [LoRA (arXiv:2106.09685)](https://arxiv.org/abs/2106.09685) · [QLoRA (arXiv:2305.14314)](https://arxiv.org/abs/2305.14314)
- [AdaLoRA: adaptive budget allocation (arXiv:2303.10512)](https://arxiv.org/abs/2303.10512) · [DyLoRA: dynamic search-free rank (arXiv:2210.07558)](https://arxiv.org/abs/2210.07558)
- [ARD-LoRA: dynamic rank allocation for heterogeneous adaptation (arXiv:2506.18267)](https://arxiv.org/abs/2506.18267) (recent; directly relevant to per-layer rank selection)

**Swarm-as-optimizer (the conceptual heart of SAGE)**
- [Model Swarms: adapt LLM experts via swarm intelligence (arXiv:2410.11163)](https://arxiv.org/abs/2410.11163) — read this closely; it's the closest published cousin to SAGE.

**Agent loop**
- [ReAct (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629) · [Reflexion (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366)

### 3.7 Minimal prototype to build (a weekend → makes "in development" real)

- **Model pool**: 3 small open models (e.g., a 1B, a 3B, a 7B) via HuggingFace.
- **Selector**: a simple scorer over (task-embedding, model-profile) — even a kNN router à la RouteLLM is enough to start.
- **Adaptation arm**: wire HuggingFace **PEFT** to run LoRA at 2–3 ranks + a QLoRA option on the sample data (a few hundred steps each).
- **Evaluator**: exact-match / loss on a held-out slice; record params + wall-clock.
- **Loop**: pick the Pareto-best (quality vs. params vs. latency); print the decision *with the proxy numbers that justified it*.
- **Grounding (stretch)**: retrieve 3 relevant arXiv abstracts per task to bias the selector. *Now SAGE is real and measurable.*

---

## 4. How HIVE and SAGE relate (tell this as one story)

HIVE is the **general-purpose swarm substrate**; SAGE is a **specialized swarm** whose task happens to be model selection + adaptation. In interviews: *"SAGE is what you get when you point a HIVE-style swarm at the AutoML problem and ground its agents in the PEFT and routing literature."* That framing makes both projects feel like one coherent research program — which is exactly the signal a hiring committee wants from a non-PhD candidate competing on depth.

---

## 5. Learning resources (verified links)

### Foundations & surveys (read these first)
- [LLM Multi-Agent Systems: Challenges and Open Problems (arXiv:2402.03578)](https://arxiv.org/abs/2402.03578)
- [A Survey on LLM-based Multi-Agent Systems (arXiv:2412.17481)](https://arxiv.org/abs/2412.17481)
- [Multi-Agent Collaboration Mechanisms: A Survey (arXiv:2501.06322)](https://arxiv.org/abs/2501.06322)
- Core patterns: [ReAct (2210.03629)](https://arxiv.org/abs/2210.03629), [Reflexion (2303.11366)](https://arxiv.org/abs/2303.11366), [Tree of Thoughts (2305.10601)](https://arxiv.org/abs/2305.10601), [Toolformer (2302.04761)](https://arxiv.org/abs/2302.04761)

### Hands-on courses
- **DeepLearning.AI — AI Agents in LangGraph** (taught by Harrison Chase, LangChain founder): https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/
- **Coursera (IBM) — Agentic AI with LangGraph, CrewAI, AutoGen & BeeAI**: https://www.coursera.org/learn/agentic-ai-with-langgraph-crewai-autogen-and-beeai
- **DataCamp — CrewAI vs LangGraph vs AutoGen** (tutorial): https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen

### YouTube (verified)
- Harrison Chase — *3 ingredients for building reliable enterprise agents*: https://www.youtube.com/watch?v=kTnfJszFxCg
- Harrison Chase — *Building Reliable AI Agents with LangGraph (Amsterdam)*: https://www.youtube.com/watch?v=DSdyqPzdlQU
- Harrison Chase — *Long-Term Memory with LangGraph*: https://www.youtube.com/watch?v=R0OdB-p-ns4
- Harrison Chase — *Context Engineering for Long-Horizon Agents*: https://www.youtube.com/watch?v=vtugjs2chdA
- **freeCodeCamp — full LangGraph course** (3-hour, free): https://www.freecodecamp.org/news/learn-langgraph-and-build-conversational-ai-with-python/
- **freeCodeCamp — How to Build Advanced AI Agents** (multi-agent, 1-hour): https://www.freecodecamp.org/news/how-to-build-advanced-ai-agents/

### Medium (verified)
- *A Developer's Guide to Multi-Agent Frameworks: CrewAI, AutoGen, LangGraph*: https://medium.com/aigenverse/a-developers-guide-to-multi-agent-frameworks-crewai-autogen-and-langgraph-15531c0c7dfe
- *10 AI Agent Frameworks You Should Know in 2026*: https://medium.com/@atnoforgenai/10-ai-agent-frameworks-you-should-know-in-2026-langgraph-crewai-autogen-more-2e0be4055556

### SAGE-specific (routing + PEFT)
- [RouteLLM (2406.18665)](https://arxiv.org/abs/2406.18665) + [LMSYS blog](https://www.lmsys.org/blog/2024-07-01-routellm/) · [AdaLoRA (2303.10512)](https://arxiv.org/abs/2303.10512) · [DyLoRA (2210.07558)](https://arxiv.org/abs/2210.07558) · [ARD-LoRA (2506.18267)](https://arxiv.org/abs/2506.18267) · [Model Swarms (2410.11163)](https://arxiv.org/abs/2410.11163)

---

## 6. A 2-week prep sprint

- **Days 1–3:** Read the two surveys (§5) + ReAct + Reflexion. Watch one Harrison Chase talk. Outcome: you can speak the agentic-AI vocabulary fluently.
- **Days 4–7:** Build the **HIVE** prototype (§2.7). Log the three metrics.
- **Days 8–10:** Read RouteLLM + AdaLoRA + DyLoRA + Model Swarms. Outcome: you can defend SAGE's selection/adaptation choices with citations.
- **Days 11–14:** Build the **SAGE** prototype (§3.7). Log Pareto results. Push both prototypes to GitHub with clean READMEs.

After this, both résumé items are literally true, you have repos to show, and you can whiteboard either system under questioning.

---

## 7. Interview talking points (the story arc)

1. **Open with the gap you saw:** "Most multi-agent systems are statically wired and fail silently."
2. **Name the move:** "HIVE makes orchestration emergent and every handoff contract-verified."
3. **Show systems judgment:** the cost tradeoff (small models for the swarm, strong model for the planner), the eval metrics (step efficiency, recovery rate).
4. **Connect to research:** "SAGE points that swarm at AutoML and grounds it in the routing + PEFT literature — Model Swarms is the closest published cousin, and I extend it with literature-grounded selection and my TARS rank-selection."
5. **Be honest about maturity:** "These are prototypes I'm actively building; here are the repos and the early numbers." Honesty + working code beats inflated claims every time — especially against PhD candidates who *will* probe.

---

*Justification note: every architectural claim above is tied to a named, citable artifact in §2.6, §3.6, and §5. Treat any performance figure as a target to measure with the §2.7 / §3.7 prototypes — do not state it as a result until you have run it.*
