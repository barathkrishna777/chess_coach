# Plan 017 - Explanation Quality Upgrade

Status: **planned**
Owner: barathkrishna
Last updated: 2026-04-16

## Summary

Improve coach-note teaching quality without weakening the existing grounding
pipeline. The explanation layer stays Stockfish-grounded: the classifier supplies
motifs, Stockfish supplies the best move and main line, the provider only
paraphrases, and invalid provider output is replaced by deterministic fallback
text.

This slice does not add image inputs, OCR, computer vision, hosted-provider
requirements, auth, cloud deployment, payments, sharing, or new product surfaces.
Ollama/local remains the default provider path.

## Key Changes

- Add motif-specific teaching guidance to `chess_ml/explanation/prompt.py` for all
  current motif IDs: allowed tactic, endgame slip, pin, fork, overloaded defender,
  discovered attack, hanging piece, missed tactic, and opening inaccuracy.
- Keep strict JSON output with `text` and `referenced_move_uci`; only display model
  output after `validate_provider_response()` accepts it.
- Keep internal engine field names such as `ground_truth_pv`, but require
  user-facing explanation text to say "main line", not "PV" or "principal
  variation".
- Moderately increase the prompt word budget while preserving the three-sentence
  limit and concise validation trimming.
- Improve deterministic fallback lessons using only existing `MotifEvidence`
  fields: attackers, defenders, piece, best_move, opponent_reply, related_ply, and
  phase.

## Test Plan

- Unit-test motif-specific prompt facts for every current motif ID.
- Assert prompts do not include image, OCR, screenshot, or computer-vision
  language.
- Unit-test fallback text for all motifs, including newer tactical motifs.
- Keep validation tests for wrong referenced moves, invented engine claims, and
  generic ungrounded advice; add rejection coverage for user-facing "PV" wording.
- Confirm cache behavior remains stable except for intentional cache-key changes
  from the added prompt facts.

Required verification before declaring done:
`make check`, `cd web && npm run typecheck`, `cd web && npm run build`, and
`make e2e`.

## Assumptions

- No API schema change is needed; prompt facts and fallback text can improve the
  coach note without changing public response models.
- `PROMPT_VERSION` remains `grounded-coach.v1`; the cache key changes through the
  structured prompt facts.
- Fallback text must remain deterministic and may only name moves, pieces, and
  tactical evidence that already exist in the supplied request.
