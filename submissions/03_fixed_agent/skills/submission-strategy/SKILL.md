---
name: submission-strategy
description: Selects best submissions based on public scores
---

# Submission Strategy Skill

Use this skill to decide which submissions to keep as final answers.

## Instructions

1. Call `get_status()` to review all submission IDs and their public scores.
2. Rank submissions by score, favoring ensembles/blends over single models when scores are close.
3. Reserve at least 2 submissions across the session for final selection.
4. Call `select_submission([...])` with the top 1-2 submission IDs before the session ends.
