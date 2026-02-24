# Improvement Plan V3: Live Site Repair

## Phase 1: Backend & Data Fixes (CRITICAL)
- [ ] **Data Service Patch**: Modify `dashboard/data_service.py` (`_get_players_from_props`) to remove the strict "DraftKings" filter. Accept ['draftkings', 'fanduel', 'rebet', 'betmgm', 'caesars'] or allow ALL.
- [ ] **Verify Local Data**: Run `test_props_unlocked.py` (or similar) locally to confirm props are now generating with "rebet" or other sources.
- [ ] **Deploy Backend**: Deploy the Python backend to a cloud provider (Railway/Render) using the existing `railway.toml`.
    - Note: If user has no Railway account, we must instruct them to set one up or use a different method. We will assume Railway for now as `railway.toml` exists.

## Phase 2: Frontend Connection
- [ ] **Update Vercel Env**: Update the `VITE_API_URL` environment variable on Vercel to point to the new Cloud Backend URL (e.g. `https://nba-backend.up.railway.app/api`).
- [ ] **Local Dev Fix**: Update `frontend/.env.local` to `VITE_API_URL=http://localhost:8000/api` (ensure `/api` suffix is present).

## Phase 3: UI/UX Polishing
- [ ] **Fix Dead Buttons**: Ensure Game Cards have `onClick` handlers that route to `/game/:id` or open a modal.
- [ ] **Empty State Handling**: Add a "Retry" button or clear error message when API fails, instead of broken UI.
- [ ] **Game Locking**: Visually gray out "TAKE" buttons for games that have started (Backend already enforces this, Frontend must reflect it).

## Phase 4: Verification
- [ ] **Live Smoke Test**: Visit `https://nba-bets.vercel.app/`, confirm picks load, buttons work, and no 404s in console.
