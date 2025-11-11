# ref-log.md

**How to run**
1. Run the below commands in the terminal and set the keys:
export OPENAI_API_KEY=""
export TAVILY_API_KEY=""
2.Run the streamlit app
streamlit run assign_2.py

**Reflection (~300 words)**

Implementing a multi‑agent workflow clarified how division of labor plus explicit interfaces can tame open‑ended tasks. By constraining the Planner to “assumptions → itinerary → budget/logistics summary” (no internet) and giving the Reviewer a fact‑checking mandate (with `internet_search`), each agent could specialize: one generates a coherent first draft, the other pressure‑tests feasibility. The orchestration—Planner → Reviewer, with tool logs visible—made the system feel both more reliable and auditable.

**Challenges & how I addressed them.**  
(1) Over‑ or under‑searching by the Reviewer-Initially the Reviewer called the tool too often or pasted long sources. I fixed this by specifying when to search (“whenever facts influence feasibility”) and requiring concise, plain‑text citations without URLs.  
(2) Budget consistency -Early drafts produced mismatched daily totals; prompting the Planner to output per‑activity costs plus a Daily Subtotal and a final summary made reconciliation easier.  
(3) Style drift in revisions-The Reviewer sometimes rewrote too aggressively. I required a “Delta List (Change → Reason)” and asked the Reviewer to preserve the Planner’s tone and constraints when producing the Revised Itinerary.

**Creative ideas / design choices.**  
I framed personas deliberately: Planner = optimistic trip designer (clusters nearby activities, inserts breaks), Reviewer = pragmatic validator (flags closures, transit windows, overbooking). Both prompts use strict output schemas so the UI can reliably show sections and so the Reviewer can “patch” the plan deterministically. I also emphasized pacing (buffers, mealtimes) and logistics (walk/metro/train durations) to keep plans realistic even without live data in the Planner stage.

**What I learned.**  
Prompt structure is product design- good sectioning and success criteria produce higher‑quality drafts than clever wording. Tool use improves trust only when it’s targeted and visible, the sidebar logs were invaluable for debugging and explaining decisions. Finally, multi‑agent handoffs are where quality is won or lost—clear contracts and diff‑like edits make collaboration work.

---

**External tools / GenAI assistance (not counted):** Used the provided `internet_search` in the Reviewer for opening hours, prices, and transit norms; leveraged an LLM to draft and tighten prompt language and section schemas.

