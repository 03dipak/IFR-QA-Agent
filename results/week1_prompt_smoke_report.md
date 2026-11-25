# Week-1 prompt baseline smoke (SMOKE_NOT_SCORED)

Dev split only; 10 rows per arm.

## B0
- row: G-001
- status: answered
- latency_ms: 401
- usage: {'prompt_tokens': 307, 'total_tokens': 318, 'completion_tokens': 11, 'prompt_tokens_details': None}
- score: {'gold_abstain': False, 'pred_abstain': True, 'abstain_correct': False, 'false_answer': False, 'false_abstention': True, 'numeric_match': None, 'citation_valid': None, 'parse_ok': True}

### Rendered prompt
```text
SYSTEM: You are a precise financial-results answerer. Closed-book: answer only from the question and the few-shot examples; never invent a figure. Output exactly one line: answer: <value> | page: <n>, or ABSTAIN.

Few-shot examples (non-gold):
Q: For the consolidated results of Aarav Pharma for the year ended 31 March 2026, what was total assets, exactly as printed, in Rs. in crore?
A: answer: 4,860 | page: 1

Q: For the consolidated results of Kaveri Logistics for the year ended 31 March 2026, what was total income, in Rs. in crore?
A: answer: 9,310 | page: 1

Q: On the consolidated statement page of Northstar Textiles for the year ended 31 March 2026, give profit before tax as printed, in Rs. in lakh.
A: answer: 1,245 | page: 1

USER: Question: Between Deepak Nitrite and Srf, on total assets for the year ended 31 March 2025 consolidated results, which prints the larger figure, and what are the two figures (in Rs. in crore, as printed on each statement)?

If the answer is not present, reply with the single word: ABSTAIN
```

## B1
- row: G-001
- status: answered
- latency_ms: 1033
- usage: {'prompt_tokens': 3019, 'total_tokens': 3060, 'completion_tokens': 41, 'prompt_tokens_details': None}
- score: {'gold_abstain': False, 'pred_abstain': False, 'abstain_correct': True, 'false_answer': False, 'false_abstention': False, 'numeric_match': True, 'citation_valid': True, 'parse_ok': True}

### Rendered prompt
```text
SYSTEM: You are a precise financial-results answerer. Use only the provided oracle context pages. Output exactly one line: answer: <value> | page: <n>, or ABSTAIN.

Oracle context (gold pages, smoke only):

USER: [document_id=fa7dac2401f10d824c6e6b3e5dd482a0e0ab09b50e6164eeebc289ec952fff31 page=363]
DEEPAK NITRITE LIMITED54th Annual Integrated Report 2024-25
360
Consolidated Balance Sheet
as at March 31, 2025
Corporate Identification Number: L24110GJ1970PLC001735
₹ in Crores
Notes 
As at
March 31, 2025
As at
March 31, 2024
I. ASSETS
Non-Current Assets
(a) Property, Plant and Equipment 2 2,400.20 2,243.11 
(b) Capital Work-in-Progress 2 1,649.05 773.51 
(c) Goodwill 2.66 - 
(d) Intangible Assets 3 54.37 49.61 
(e) Financial Assets
Investments 4 2.98 2.69 
Loans 5 1.96 2.09 
Other Financial Assets 6 15.52 28.60 
(f) Non-Current Tax Assets (Net) 7 0.40 4.86 
(g) Deferred Tax Assets 8 11.65 2.17 
(h) Other Non-Current Assets 9 281.77 147.98 
Total Non-Current Assets 4,420.56 3,254.62 
Current Assets
(a) Inventories 10 926.40 759.91 
(b) Financial Assets
Investments 11 507.91 119.16 
Trade Receivables 12 1,273.81 1,298.35 
Cash and Cash Equivalents 13 179.41 237.97 
Bank balances other than Cash and Cash Equivalents above 14 227.15 227.49 
Loans 5 1.74 1.11 
Other Financial Assets 15 18.56 62.21 
(c) Other Current Assets 16 162.13 135.40 
Total Current Assets 3,297.11 2,841.60 
TOTAL ASSETS 7,717.67 6,096.22 
II. EQUITY AND LIABILITIES
Equity
(a) Equity Share Capital 17 27.28 27.28 
(b) Other Equity 18 5,361.39 4,769.34 
(c) Non-Controlling Interest 35.99 26.06 
Total Equity 5,424.66 4,822.68 
Liabilities
Non-Current Liabilities
(a) Financial Liabilities
Borrowings 19 1,025.71 216.95 
Lease Liabilities 43 88.60 62.28 
(b) Provisions 20 43.16 35.63 
(c) Deferred Tax Liabilities (Net) 21 212.78 173.62 
Total Non-Current Liabilities 1,370.25 488.48 
Current Liabilities
(a) Financial Liabilities
Borrowings 22 144.82 - 
Lease Liabilities 43 7.88 6.85 
Trade Payables
Total out
```
