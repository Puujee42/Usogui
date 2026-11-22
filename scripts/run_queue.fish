#!/usr/bin/env fish
# Sequential training queue for the Frege detector ladder.
# Runs: CE frege seeds 1337, 2026; then format ablation data_only and plain_inst_data (seed 42).
# Each run logs to reports/<name>.log. The queue stops at the first failure.
# Preflight: the three audit commands (CLAUDE.md section 8) must pass before any training starts.
#
# Usage (from anywhere):  fish scripts/run_queue.fish

set -l root (realpath (dirname (status filename))/..)
cd $root; or exit 1
set -l py $root/.venv/bin/python

function stamp
    date '+%Y-%m-%d %H:%M:%S'
end

echo "["(stamp)"] preflight: audits"
$py -m pytest tests -q -rs; or begin; echo "["(stamp)"] STOP: pytest failed"; exit 1; end
$py tests/mutation_check.py; or begin; echo "["(stamp)"] STOP: mutation_check failed"; exit 1; end
$py src/12_verify_agent_claims.py; or begin; echo "["(stamp)"] STOP: 12_verify_agent_claims failed"; exit 1; end

# name  format  seed
set -l jobs \
    "ce_frege_seed1337 frege 1337" \
    "ce_frege_seed2026 frege 2026" \
    "ce_data_only_seed42 data_only 42" \
    "ce_plain_inst_data_seed42 plain_inst_data 42"

for job in $jobs
    set -l f (string split ' ' -- $job)
    set -l name $f[1]
    set -l fmt $f[2]
    set -l seed $f[3]
    set -l log reports/$name.log

    echo "["(stamp)"] START $name (format=$fmt seed=$seed) -> $log"
    $py -u src/22_train_ce.py --format $fmt --seed $seed 2>&1 | tee $log
    set -l rc $pipestatus[1]
    if test $rc -ne 0
        echo "["(stamp)"] STOP: $name failed with exit code $rc (see $log)"
        exit $rc
    end
    echo "["(stamp)"] DONE  $name"
end

echo "["(stamp)"] queue finished: all 4 runs completed"
