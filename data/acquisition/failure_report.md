# Failure Report — Acquisition Blockers (read-only)

All entries below are derived from existing artifacts (`data/seeds/company_domains.csv`,
`data/seeds/manual_collection_list.csv`, `data/seeds/coverage_matrix.csv`,
`data/seeds/harvest_p12.log`, `data/seeds/final_push.log`). No new network requests were made.

## 1. Priority-3 blocked companies (12 companies × 13 missing cells = 156 cells unavailable)

| Company | Sector | Domain status | Reason | Assessment under current constraints |
|---|---|---|---|---|
| alkyl_amines | chemicals | UNVERIFIED | domain UNREACHABLE | unfixable without reachable domain |
| balkrishna | auto | UNVERIFIED | domain UNREACHABLE (3 alternate guesses also failed) | unfixable without reachable domain |
| cg_power | capital goods | UNVERIFIED | domain UNREACHABLE | unfixable without reachable domain |
| endurance | auto | UNVERIFIED | domain NO_INVESTOR_LINK (reachable, no investor href) | mitigable only if investor path discovered |
| fine_organics | chemicals | UNVERIFIED | domain UNREACHABLE | unfixable without reachable domain |
| godrej_consumer | FMCG | UNVERIFIED | domain UNREACHABLE | unfixable without reachable domain |
| havells | capital goods | UNVERIFIED | domain NO_INVESTOR_LINK | mitigable only if investor path discovered |
| kei | capital goods | UNVERIFIED | domain UNREACHABLE | unfixable without reachable domain |
| nmdc | energy/metals | UNVERIFIED | domain NO_INVESTOR_LINK | mitigable only if investor path discovered |
| pi_industries | chemicals | UNVERIFIED | domain UNREACHABLE | unfixable without reachable domain |
| vinati | chemicals | UNVERIFIED | domain NO_INVESTOR_LINK | mitigable only if investor path discovered |
| zydus_life | pharma | UNVERIFIED | domain UNREACHABLE (2 alternate guesses failed) | unfixable without reachable domain |

## 2. Verified-domain companies with zero discovered document links (8 companies)

| Company | Sector | Domain | Issue |
|---|---|---|---|
| divis_labs | pharma | www.divislabs.com VERIFIED | no static PDF links found on discovered pages |
| abbott_india | pharma | www.abbott.in VERIFIED | no static PDF links found (December year-end) |
| britannia | FMCG | www.britannia.co.in VERIFIED | 1 listing page, 0 PDFs matched |
| ntpc | energy/metals | www.ntpc.co.in VERIFIED | annual candidates 404; no static quarterly links |
| tech_mahindra | IT | www.techmahindra.com VERIFIED | 17 ambiguity rows, no unique statement-like URL resolved |
| marico | FMCG | www.marico.com VERIFIED | no static PDF links found |
| force_motors | auto | www.forcemotors.com VERIFIED | no static PDF links found |
| adani_green | energy/metals | www.adani.com VERIFIED | known pages are newsroom/media releases, not financial filings |

These are mitigable only if deeper IR sections (results pages, paginated archives) yield
static PDF links; none were found within the executed harvest bounds.

## 3. What is NOT fixable under current constraints

- NSE/BSE bulk automation (rule 3): requires explicit permission or mentor/institution
  academic access — treated as pending, not attempted.
- Unreachable domains (8 companies): no bypass per rules; re-test only if domain becomes
  reachable.
- JavaScript-rendered listing pages: no browser spoofing per rules.

## 4. Open items for the pilot phase

- Re-check the 4 NO_INVESTOR_LINK domains (havells, nmdc, vinati, endurance) for
  investor-path discovery results from any subsequent harvest.
- Resolve 71 ambiguous cells across 227 ambiguity rows (tech_mahindra 39, eicher_motors
  29, polycab 25, itc 20, hindalco 17, ashok_leyland 16) where multiple candidate URLs
  exist but none could be singled out as THE statement document.
