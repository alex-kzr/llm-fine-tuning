# Change Request (CR) Process

This project uses a lightweight, file-based process for proposing and tracking changes.

## When to create a Change Request

Create a CR when you want to:
- change the output JSON schema
- add new dataset domains or generators
- change training/inference algorithms
- modify CLI behavior or configuration keys
- introduce new dependencies or hardware support

## Where changes are tracked

- `board/board.md` — high-level Kanban status
- `board/plan.md` — phased plan and acceptance criteria
- `board/tasks/` — detailed tasks

## CR template

Copy the template below into a new file under `board/tasks/` (or use your team’s preferred tracking system), then link it from the board.

```markdown
# CR-XX - <Short title>

## Motivation
<What problem are we solving? Who benefits?>

## Proposal
<What will change? Include examples.>

## Impact
- Backward compatibility:
- Data/model artifacts impacted:
- Config/CLI changes:
- Performance/resource impact:

## Acceptance Criteria
- [ ] ...
- [ ] ...

## Risks / Open questions
- ...

## Docs to update
- docs/project-overview.md
- docs/architecture.md
- docs/domain-model.md
- docs/algorithms.md
- docs/usage-guide.md
```

## Definition of done for CRs

- Code changes implemented and validated.
- Documentation updated (at least: overview + usage).
- Example commands still work (or are updated).

