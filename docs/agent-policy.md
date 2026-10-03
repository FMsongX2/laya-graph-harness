# Cheap selection first, focused agent review when needed

1. Specify the task, source excerpt and paper scope from actual source metadata.
2. Let code follow deterministic exact citation links first.
3. Otherwise let Laya choose among the bounded live candidate menu and collect
   the selected assertion's evidence/conditions in the same request.
4. Evaluate whether the returned evidence addresses the goal and conditions.
5. If evidence is insufficient, conflicting, out of scope, or the menu missed
   the useful relationship, do a targeted agent-led search with the missing
   concept or condition. Expand the scope only when that observation warrants it.

`needs_review` is not an instruction to automatically redo the whole graph
search. Low confidence can be resolved by inspecting returned alternatives and
actual source evidence. High confidence cannot prove that the chosen path is
relevant. Valid node IDs and existing quotations also cannot establish goal
completion: all live test outputs had real evidence, but only 124/150 matched
the labeled relationship.

This policy is executed by the calling agent/host. The local service exposes
structured candidates, evidence and reasons; it does not autonomously invoke
a paid agent or execute arbitrary suggested tools. Broader goal traversal,
candidate recall, unseen field queries and learned stopping need their own
evaluation before expanding the current task profile.
