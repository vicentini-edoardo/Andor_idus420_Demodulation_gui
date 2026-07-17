# Safe GUI Updates Design

## Goal

Make the Andor iDus GUI and Red Pitaya TTL divider updater follow the same safe Git workflow while retaining an explicit restart choice.

## Behavior

1. Fetch and prune `origin` before presenting remote branches.
2. Let the user select a remote branch.
3. Check out its local branch, creating it as a tracking branch when it does not exist.
4. Set that branch's upstream to the selected remote branch.
5. Pull using `--ff-only`.
6. Compare `HEAD` before and after the operation and show the new commit subjects when it changed.
7. Offer `Restart now` or `Later`; never restart automatically.

## Boundaries

- Keep each application's current Qt UI and background-execution pattern.
- Add only small testable Git helpers and focused tests; no cross-repository shared package.
- Preserve Red Pitaya's legacy `rp_state.json` cleanup.

## Failure handling

Git command failures are reported in the existing UI and do not restart the application. A fast-forward-only pull rejects merge-required updates rather than creating a merge commit.

## Verification

Focused unit tests cover branch parsing, creation/tracking of a missing local branch, fast-forward pull, change detection, and no-change behavior. Run the affected test suites and Ruff for both repositories.
