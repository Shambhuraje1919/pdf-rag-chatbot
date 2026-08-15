# RAG Accuracy Testing

This folder documents how the chatbot's retrieval and citation accuracy were tested.

## What's here

- **TechNova_Employee_Handbook.pdf** — a 10-page synthetic policy document used as the test corpus. Each page covers one distinct section (password policy, data classification, remote work, leave, expense reimbursement, etc.), so retrieved answers can be checked against an exact, known section and page.
- **personalized_results.csv** — the actual test log: 16 questions run against the deployed chatbot, covering single-chunk factual questions, multi-hop questions (answer spans two sections), and abstention questions (answer is deliberately not in the document, to test that the bot says so instead of hallucinating).
- **RAG_Accuracy_Report.pdf** — the generated report summarizing results: 100% answer accuracy and 100% citation accuracy across all 16 tested questions.

## Method

Each question was asked directly to the deployed Streamlit app. The chatbot's answer and its reported citation (section + page) were logged and manually checked against the document's actual content. Two categories were tested:

1. **Single-chunk / multi-hop questions** — graded correct only if both the answer and the cited section/page matched the ground truth.
2. **Abstention questions** — graded correct only if the bot explicitly stated the answer wasn't in the document, rather than fabricating one.

## A bug found during testing

Early testing caught a real citation bug: page numbers were stored 0-indexed (from `PyPDFLoader`) and section names were never set in the document metadata, so the chatbot's cited section silently fell back to whatever the LLM guessed from the visible chunk text — correct only by coincidence. This was fixed by adjusting the ingestion step to store 1-indexed page numbers and extract an actual section heading into the metadata. Results in this folder reflect the app **after** that fix.

## Notes

This was a manually run, 16-question test, covering single-chunk retrieval, multi-hop reasoning, and abstention on out-of-scope questions.
