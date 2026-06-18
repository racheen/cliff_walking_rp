# Global Codex Instructions for `cliff_walking_rp`

## Scope

This file applies to the whole repository.

## Working Style

- Inspect existing project patterns before changing code.
- Make the smallest correct change that solves the request.
- Preserve current behavior and data shapes unless explicitly asked to change them.
- Leave unrelated dirty-worktree edits alone.
- Prefer Conventional Commits for any commit messages.

## Project Shape

- `site/` contains the Next.js viewer.
- The Python training/export pipeline lives at the repo root and in `src/`.
- Static run artifacts are served from `site/public/runs/`.

## UI Direction

- Default to a neo-minimal look.
- Use a soft pastel green palette with restrained contrast.
- Keep spacing generous and surfaces subtle.
- Preserve accessibility and responsive behavior.

## Documentation

- Update `project.md` when the project shape or main workflows change.
- Keep repo guidance concise and practical for future Codex runs.

