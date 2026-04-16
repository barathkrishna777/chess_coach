# Plan 018 — Productization And Public GTM Roadmap

Status: **proposed, not implemented**  
Owner: barathkrishna  
Last updated: 2026-04-16  
Target path: `docs/plans/018-productization-public-gtm.md`

## Summary

Build toward a hosted SaaS release for individual 1200–2000 rated players, while preserving the local-first product and all MVP constraints. The wedge is not generic cloud chess analysis; it is a personal coach loop: user games → recurring weakness diagnosis → grounded explanation → own-game drills → measurable improvement.

This plan starts after the current local product foundation: PGN review, Stockfish grounding, local play, dashboard, openings, personal drills, richer motifs, learned classifier path, and explanation validation.

Priority order: first deepen the coaching value, then add multi-user/cloud foundations, then monetize and share only the parts that strengthen go-to-market.

## Priority Roadmap

1. **Personal Coach Plan v1**
   - Add a weekly improvement plan generated from stored reviews, drill history, motifs, phases, and openings.
   - Rank the top 3 weaknesses by impact, recurrence, and recency.
   - Each recommendation links to evidence from the user’s own games and a concrete drill queue.
   - API: `GET /api/coach/plan`, `POST /api/coach/plan/refresh`, `POST /api/coach/tasks/{task_id}/complete`.

2. **Retention-Grade Training Loop**
   - Upgrade `/train` from single-position drills into sessions: 5–10 positions, completion state, streaks, and progress by motif.
   - Keep own-game positions as the default source.
   - Add external puzzle databases only as motif-matched reinforcement when the user lacks enough own-game positions.
   - Store puzzle source as `own_game` or `external_matched`; do not add a generic puzzle browser.

3. **Multi-User Auth And Data Isolation**
   - Add provider-neutral OIDC/JWT auth.
   - Add `GET /api/me`.
   - Scope all profile, review, drill, coach-plan, explanation-cache, and billing reads/writes by `user_id`.
   - Local mode keeps `make serve` working with a deterministic `local-user`.
   - Existing local SQLite rows migrate to `local-user`.

4. **Hosted Data Layer**
   - Introduce a database access layer using SQLAlchemy Core plus Alembic migrations.
   - Keep SQLite for local development and Postgres for hosted deployment via `DATABASE_URL`.
   - Add `user_id` to user-owned tables and enforce unique constraints such as `(user_id, game_id)`.
   - Preserve deterministic review IDs and idempotent re-analysis behavior.

5. **Hosted Review Jobs And Deployment**
   - Add async hosted review flow:
     - `POST /api/reviews` creates a job from PGN and returns `review-job.v1`.
     - `GET /api/reviews/{review_id}` returns queued/running/succeeded/failed plus result when ready.
   - Keep existing synchronous `/api/games` for local/dev compatibility.
   - Deploy provider-neutrally as Docker images: `web`, `api`, and `worker`.
   - Use a Postgres-backed job table first; defer Redis/Celery/Kubernetes until real queue pressure exists.
   - Worker owns Stockfish analysis; explanations remain lazy/on-demand.

6. **Hosted LLM Quality And Cost Testing**
   - Keep the grounding pipeline unchanged: Stockfish and motifs are truth, LLMs only teach.
   - Add provider adapters behind the existing explanation service contract.
   - Add an eval harness over fixed positions to measure validity, latency, and cost per accepted coach note.
   - Hosted provider failures must fall back to deterministic grounded text and must not block review.

7. **Payments And Entitlements**
   - Add provider-neutral billing interfaces, with adapter-specific implementation isolated.
   - API: `GET /api/billing/status`, `POST /api/billing/checkout`, `POST /api/billing/portal`, `POST /api/billing/webhook`.
   - Default tiers:
     - `free`: limited hosted reviews per month, dashboard, own-game drills, deterministic fallback notes.
     - `plus`: higher review limits, hosted coach notes, long-term coach plan history.
   - Entitlements gate hosted capacity and paid coach features, not local `make serve`.

8. **Shareable Reports, Not Social Network**
   - Add revocable read-only share links for one reviewed game or one coach-plan report.
   - API: `POST /api/share/games/{game_id}`, `POST /api/share/reports/{report_id}`, `DELETE /api/share/{share_id}`.
   - Shared pages expose only sanitized review/plan data selected by the owner.
   - No feeds, likes, comments, follows, or public dashboards in the first GTM release.

9. **Public Beta Launch Readiness**
   - Add onboarding that gets a new user to value in one session: sample game, PGN upload, first coach plan, first drill session.
   - Add privacy-forward product copy: PGNs are user data, no image input, no engine-contradicting advice.
   - Add minimal product telemetry without storing raw PGN in analytics events.
   - Add admin/support visibility for job failures, provider failures, and payment status.

10. **Later Strategic Bets**
   - Opponent prep/style similarity comes after retention is proven; it is a tournament-prep expansion, not the first wedge.
   - Cloud-scale infra/concurrency upgrades happen after job-table limits are observed.
   - Large from-scratch chess models remain out of scope unless the small classifier plus Stockfish-grounded pipeline demonstrably cannot support the product.

## Key Contracts

- `auth-user.v1`: `{ user_id, email, display_name, created_at, plan }`.
- `coach-plan.v1`: `{ generated_at, primary_goal, weaknesses[], tasks[], evidence_games[], progress }`.
- `review-job.v1`: `{ review_id, status, created_at, updated_at, error, annotated_game }`.
- `billing-status.v1`: `{ tier, entitlements, period_end, usage }`.
- All new public models include `schema_version`.
- All hosted user-owned routes require auth; local mode injects `local-user`.

## Test Plan

- Unit tests for coach-plan ranking, drill-session scheduling, entitlement gates, auth claims parsing, and user-scoped queries.
- Migration tests for SQLite local rows → `local-user` and Postgres-compatible schema creation.
- API tests proving one user cannot read another user’s games, drills, explanations, shares, or billing state.
- Job tests for queued/running/succeeded/failed review states, idempotent duplicate PGN uploads, worker retry behavior, and Stockfish unavailable failures.
- Billing tests for webhook idempotency, entitlement changes, canceled subscriptions, and missing-provider local mode.
- Playwright e2e: sign in, upload PGN, wait for hosted review job, view coach plan, complete a drill session, check billing status, create/revoke a share link.
- Required verification before declaring each slice done: `make check`, `cd web && npm run typecheck`, `cd web && npm run build`, and `make e2e`.

## Assumptions

- First buyer: individual club players, not coaches or tournament-prep teams.
- First public release shape: hosted SaaS, with local mode preserved.
- Vendor policy: provider-neutral adapters for auth, billing, deployment, and hosted LLMs.
- No image upload, OCR, screenshots, or computer vision are ever added.
- Explanations must never contradict Stockfish; invalid hosted LLM output is rejected exactly like local output.
- Generic social features, generic puzzle browsing, large model training, and production-scale infra are intentionally deferred.
