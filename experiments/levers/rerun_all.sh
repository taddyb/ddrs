#!/usr/bin/env bash
# Re-run every analysis with the final common.py (share up/down with a 1e-9 threshold). Outputs go to the scripts' own
# .txt files; this log records exit codes.
cd /home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/levers
PY=/home/tbindas/projects/ddr/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run() { echo "$(date -u +%FT%TZ) start $*" >> rerun.log; "$PY" "$@" > /dev/null 2>> rerun.err; echo "$(date -u +%FT%TZ) done $* exit=$?" >> rerun.log; }
run volume.py 42 2026-09-30T13-13-59Z-train-and-test
run volume.py 43 2026-09-30T13-50-59Z-train-and-test
run timing.py 42 2026-09-30T13-13-59Z-train-and-test
run timing.py 43 2026-09-30T13-50-59Z-train-and-test
run volume_state.py 42 2026-09-30T13-13-59Z-train-and-test
run volume_state.py 43 2026-09-30T13-50-59Z-train-and-test
run volume_by_year.py 2026-09-27T07-29-47Z-train-and-test 2026-09-30T13-13-59Z-train-and-test
run volume_by_year.py 2026-09-27T10-31-30Z-train-and-test 2026-09-30T13-50-59Z-train-and-test
run variance.py
run variance_gamma.py
run store_ensembles.py
run ens_detail.py
run store_selection.py
run combine_ens.py
run combine.py s42=2026-09-27T07-29-47Z-train-and-test:2026-09-30T13-13-59Z-train-and-test s43=2026-09-27T10-31-30Z-train-and-test:2026-09-30T13-50-59Z-train-and-test g0=2026-09-12T03-53-34Z-train-and-test:- nonly=2026-09-12T23-39-06Z-train-and-test:-
echo "$(date -u +%FT%TZ) all done" >> rerun.log
