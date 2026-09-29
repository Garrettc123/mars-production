# LLM fleet sync — 2026-09-29T06:07Z

Source of truth for every model touching Garcar / MARS.

## Shipping state
- Repo: `Garrettc123/mars-production`
- `main` SHA after squash: `90555456fd5d9e7e5b7ff77fcd88ba9f30fd3dd1`
- PR: https://github.com/Garrettc123/mars-production/pull/7
- API version: `1.1.0`

## Contract
- `POST /api/reason` uses `mars_router.run` (auto/standard/deep).
- Response includes `route` with path, reason, confidence, uncertainty, escalated.
- `GET /api/status` exposes models + threshold + `score_version` `conf-re-v1.1`.
- `GET /api/routing/stats` is the spend meter.
- Confidence line parser accepts markdown: `**confidence**: 0.91` and `confidence **:** 0.4`.

## Revenue rules (do not drift)
- Livemode Stripe `acct_1SS3dpFKGbk21LK5`.
- Default sell: LLA-47 https://buy.stripe.com/3cI00j7YV0hQgDp8BR43S2v
- Cash = Stripe charge `paid=true`. HubSpot appointment amount is not cash.
- No live 4242.

## Host action still open
Redeploy so Docker copies `mars_router.py`. Confirm `/api/status` version is 1.1.0.
