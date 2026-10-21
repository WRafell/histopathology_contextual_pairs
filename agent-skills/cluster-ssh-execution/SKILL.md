---
name: cluster-ssh-execution
description: Run project checks, analyses, and experiments from a local coding agent by using configured SSH hosts as transport to managed Slurm clusters. Use whenever a task needs remote files, environments, GPUs, job submission, or job monitoring.
---

# Cluster SSH Execution

Keep the Codex or Claude Code process on the local workstation. Never start,
install, resume, or invoke a coding agent on a remote host, including through an
SSH terminal, an IDE Remote SSH extension, or a desktop application's SSH
environment.

## Resolve the Target

The cluster target must be an SSH host alias supplied by the user or the current
project's instructions and resolvable through the local SSH configuration.
There is no default server. Do not invent a hostname, use a raw IP address, or
select a cluster because it appears idle. If remote work is required and no
configured alias is available from the task or project, ask the user for one.

Treat the repository containing the current task as the project. Resolve its
local root with `git rev-parse --show-toplevel` when it is a Git repository,
otherwise use the current working directory. There is no default project name
or remote path. Resolve the remote worktree for this same project from the
user's request or project instructions. If neither defines it, ask rather than
guessing a path from another project.

Before the first remote operation, verify the chosen alias locally and confirm
that the resolved remote directory is a Git worktree for the current project:

```bash
ssh -G "$CLUSTER_ALIAS" >/dev/null
ssh "$CLUSTER_ALIAS" "hostname; id -un; git -C '$REMOTE_PROJECT_ROOT' rev-parse --show-toplevel"
```

Use task-specific variable names such as `CLUSTER_ALIAS` and
`REMOTE_PROJECT_ROOT`; resolve both before composing later commands. A valid SSH
alias does not establish that an arbitrary remote path is the current project,
so verify its Git remote, repository name, or expected project files when
needed.

Do not open an interactive remote shell for agent work.

## Separate Control Work from Compute Work

Use direct one-shot SSH commands on the login node only for brief control-plane
operations: inspecting Git state, querying Slurm, submitting jobs, and reading
small logs. Do not run Python programs, tests, training, evaluation, data
processing, or sustained monitoring directly on a login node.

Run compute work through Slurm, using the current project's documented batch
entry point and environment setup. A typical authorized submission has this
shape:

```bash
ssh "$CLUSTER_ALIAS" "cd '$REMOTE_PROJECT_ROOT' && sbatch <project-submit-script> <arguments>"
```

Use a scheduler-backed alternative such as an appropriate `sbatch --wrap`
only when the project's launcher cannot express the command. Preserve its
documented environment and Slurm log conventions.

Keep every infrastructure-specific choice in batch-submission shell files.
Python source, method constructors, ordinary training/analysis scripts, and
experiment configuration files must not name this cluster, its nodes,
partitions, QoS, Slurm variables, GPU layout, or remote filesystem paths. The
same experiment config must remain runnable locally and on another scheduler;
only the submission layer supplies machine allocation.

## Choose QoS Before Resources

At the start of work whose resource needs are not already frozen, inspect the
configured limits:

```bash
ssh "$CLUSTER_ALIAS" 'sacctmgr show qos format=Name,MaxWall,MaxTRES,MaxTRESPU,GrpTRES,Priority'
```

Select only a QoS and resource combination reported by that cluster or required
by the current project's launcher. Honor a task's frozen resource protocol when
it is compatible with current limits; otherwise stop and report the conflict
rather than silently changing the protocol.

Check current capacity with `sinfo`, `squeue`, and, when needed,
`scontrol show job` or `scontrol show node`. Capacity is observational context,
not permission to launch.

## Synchronize Code Deliberately

Remote commands do not see uncommitted local edits automatically. Before a run,
compare the local and remote branch, commit, and worktree state. Treat saved
result artifacts and completed settings records on the server as authoritative.

- Prefer the repository's Git remote for transferring committed source.
- Ask before pushing, pulling, copying files, or otherwise mutating the remote
  working tree unless the user's current request explicitly authorizes that
  exact operation.
- Update a clean remote tree only with a fast-forward-safe workflow.
- If the remote tree is dirty or diverged, stop and report it. Never force,
  reset, clean, stash, or overwrite another worktree automatically.
- Never use `rsync --delete` against the shared checkout. Do not copy local
  environments, credentials, datasets, result records, or `.git` metadata.
- Confirm the remote commit or source fingerprint used by a submitted job and
  report it with the job ID.

Follow the current project's experiment-management instructions, when present,
before touching any file a running job may depend on.

## Preserve Authorization Boundaries

Read-only connection, repository-state, and scheduler checks are allowed when
needed for the task. Obtain user approval before an external mutation that the
request did not already authorize, including syncing code, submitting a job,
cancelling a job, changing a scheduler queue, or deleting remote artifacts.

Never cancel, requeue, resubmit, or broaden resources merely because a job is
slow or pending. Never place credentials, private keys, access tokens, or SSH
configuration in the repository or send them to a remote command.

After submission, use one-shot `squeue` or `sacct` queries to report the exact
job state. Retrieve results without deleting the server copies, and validate
completed records according to the experiment-management skill before treating
them as evidence.
