#!/usr/bin/env bash
set -euo pipefail

# Autoresearch Robotics Detached Runner
# Operates independently without any version control tools.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

LOCK_FILE="lab/run.lock"
LEDGER_FILE="lab/ledger.tsv"
PROTECTED_HASH="lab/protected.sha256"
PROTECTED_BACKUP="lab/protected_backup"
BEST_DIR="lab/best"
BEST_SRC="lab/best/src"
BEST_METRIC="lab/best/metric.json"

mkdir -p lab/exp lab/best/src lab/sessions "${PROTECTED_BACKUP}"

# Ensure ledger exists with header
if [ ! -f "${LEDGER_FILE}" ]; then
    printf "id\tts(YYYY-MM-DDTHH:MM)\ttag\tstatus\tmetric\tmetric_name\tstd\tvalid\tsecondary\tparent\tdiff\tconstraints\tdesc\n" > "${LEDGER_FILE}"
fi

# Ensure protected hash manifest exists
if [ ! -f "${PROTECTED_HASH}" ]; then
    find protected/ -type f ! -path '*/__pycache__*' | sort | xargs sha256sum > "${PROTECTED_HASH}"
    rsync -a --delete --exclude='__pycache__' protected/ "${PROTECTED_BACKUP}/"
fi

cmd="${1:-}"

case "${cmd}" in
    start)
        tag="${2:-}"
        desc="${3:-}"
        if [ -z "${tag}" ] || [ -z "${desc}" ]; then
            echo "Usage: $0 start <tag> \"<description>\"" >&2
            exit 1
        fi

        # Check pending lock
        if [ -f "${LOCK_FILE}" ]; then
            source "${LOCK_FILE}"
            if [ -n "${LOCK_PID:-}" ] && kill -0 "${LOCK_PID}" 2>/dev/null; then
                echo "Error: Experiment ${LOCK_ID:-?} is already running (PID ${LOCK_PID})." >&2
                echo "Use '$0 status' or '$0 wait'." >&2
                exit 1
            else
                echo "Notice: Removing stale lock file from PID ${LOCK_PID:-unknown}." >&2
                rm -f "${LOCK_FILE}"
            fi
        fi

        # Determine experiment ID
        exp_id=$(python lab/runner_helper.py next_id)
        exp_dir="lab/exp/${exp_id}"
        mkdir -p "${exp_dir}"

        # Snapshot current src/
        tar -czf "${exp_dir}/src.tgz" src

        # Read timeout from GOAL.md
        hard_timeout=$(python lab/runner_helper.py timeout)

        # Prepare runner command with systemd-run or ulimit fallback
        job_log="${exp_dir}/job.log"
        if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope true >/dev/null 2>&1; then
            RUN_CMD="systemd-run --user --scope -p MemoryMax=105G -p MemorySwapMax=0 timeout --kill-after=10s ${hard_timeout}s python protected/evaluate.py"
        else
            echo "Warning: systemd-run not available; using ulimit fallback" >&2
            RUN_CMD="ulimit -v 110100480 2>/dev/null || true; timeout --kill-after=10s ${hard_timeout}s python protected/evaluate.py"
        fi

        # Launch detached under setsid nohup
        setsid nohup bash -c "${RUN_CMD}" > "${job_log}" 2>&1 &
        pid=$!

        # Record lock
        cat <<EOF > "${LOCK_FILE}"
LOCK_ID="${exp_id}"
LOCK_TAG="${tag}"
LOCK_DESC="${desc}"
LOCK_PID="${pid}"
LOCK_START="$(date +%s)"
EOF

        echo "Started experiment ${exp_id} (${tag}) detached (PID ${pid})."
        ;;

    status)
        if [ -f "${LOCK_FILE}" ]; then
            source "${LOCK_FILE}"
            if [ -n "${LOCK_PID:-}" ] && kill -0 "${LOCK_PID}" 2>/dev/null; then
                now=$(date +%s)
                elapsed=$((now - ${LOCK_START:-now}))
                echo "RUNNING: ID=${LOCK_ID} TAG=${LOCK_TAG} PID=${LOCK_PID} ELAPSED=${elapsed}s"
            else
                echo "IDLE (stale lock cleared)"
                rm -f "${LOCK_FILE}"
            fi
        else
            echo "IDLE"
        fi
        ;;

    wait)
        if [ ! -f "${LOCK_FILE}" ]; then
            echo "No job running."
            exit 0
        fi

        source "${LOCK_FILE}"
        exp_id="${LOCK_ID}"
        tag="${LOCK_TAG}"
        desc="${LOCK_DESC}"
        pid="${LOCK_PID}"
        exp_dir="lab/exp/${exp_id}"
        job_log="${exp_dir}/job.log"

        # Poll up to 240 seconds (~4 minutes)
        poll_count=0
        while kill -0 "${pid}" 2>/dev/null; do
            sleep 1
            poll_count=$((poll_count + 1))
            if [ ${poll_count} -ge 240 ]; then
                echo "STILL RUNNING: ID=${exp_id} PID=${pid}"
                if [ -f "${job_log}" ]; then
                    echo "--- Recent log tail ---"
                    tail -n 10 "${job_log}"
                fi
                exit 0
            fi
        done

        # Process has completed. Follow decision sequence in spec:

        # 1. Tamper check
        tampered=0
        if ! sha256sum --status -c "${PROTECTED_HASH}" 2>/dev/null; then
            tampered=1
            echo "Tamper check failed! Restoring protected/ from backup..."
            rsync -a --delete "${PROTECTED_BACKUP}/" protected/
        fi

        # 2. Diff generation before any restore of src/
        diff_patch="${exp_dir}/diff.patch"
        if [ -d "${BEST_SRC}" ] && [ "$(ls -A "${BEST_SRC}")" ]; then
            diff -ruN "${BEST_SRC}" src > "${diff_patch}" 2>/dev/null || true
        else
            # First baseline: compare against empty directory
            empty_tmp=$(mktemp -d)
            diff -ruN "${empty_tmp}" src > "${diff_patch}" 2>/dev/null || true
            rmdir "${empty_tmp}"
        fi

        # Truncate diff to first 300 lines
        head -n 300 "${diff_patch}" > "${diff_patch}.tmp" && mv "${diff_patch}.tmp" "${diff_patch}"

        # Compute diff stats (f<files>+<added>-<removed>)
        files_count=$(grep -c "^--- " "${diff_patch}" 2>/dev/null || true)
        if [ -z "${files_count}" ] || [ "${files_count}" -eq 0 ]; then
            # If no file header matched, check for empty diff
            if [ ! -s "${diff_patch}" ]; then
                files_count=0
            else
                files_count=1
            fi
        fi
        lines_add=$(grep -E "^\+[^+]" "${diff_patch}" 2>/dev/null | wc -l || true)
        lines_rem=$(grep -E "^-[^-]" "${diff_patch}" 2>/dev/null | wc -l || true)
        diff_summary="f${files_count}+${lines_add}-${lines_rem}"

        # Determine parent ID
        parent_id="000"
        if [ -f "${BEST_METRIC}" ]; then
            parent_id=$(python -c "import json; print(json.load(open('${BEST_METRIC}'))['id'])" 2>/dev/null || echo "000")
        fi

        # Save log tail (last 200 lines)
        tail -n 200 "${job_log}" > "${exp_dir}/log_tail.txt"

        # Handle tamper status immediately
        if [ "${tampered}" -eq 1 ]; then
            # Restore src if best exists
            if [ -d "${BEST_SRC}" ] && [ "$(ls -A "${BEST_SRC}")" ]; then
                rsync -a --delete "${BEST_SRC}/" src/
            fi
            echo "{\"valid\": 0, \"invalid_reason\": \"Tamper check failed\"}" > "${exp_dir}/result.json"
            python lab/runner_helper.py append_ledger \
                --id "${exp_id}" --tag "${tag}" --status "tampered" --metric "nan" \
                --metric-name "unknown" --std "0.0" --valid 0 --secondary "" \
                --parent "${parent_id}" --diff "${diff_summary}" --constraints "tampered" \
                --desc "${desc}"
            rm -f "${LOCK_FILE}"
            echo "TAMPERED: Run discarded and protected/ restored."
            exit 0
        fi

        # 3. Check for RESULT line
        result_line=$(grep "^RESULT " "${job_log}" | tail -n 1 || true)
        if [ -z "${result_line}" ]; then
            # Crash: no RESULT line
            if [ -d "${BEST_SRC}" ] && [ "$(ls -A "${BEST_SRC}")" ]; then
                rsync -a --delete "${BEST_SRC}/" src/
            fi
            echo "{\"valid\": 0, \"invalid_reason\": \"No RESULT line produced (crash)\"}" > "${exp_dir}/result.json"
            python lab/runner_helper.py append_ledger \
                --id "${exp_id}" --tag "${tag}" --status "crash" --metric "nan" \
                --metric-name "unknown" --std "0.0" --valid 0 --secondary "" \
                --parent "${parent_id}" --diff "${diff_summary}" --constraints "crash" \
                --desc "${desc}"
            rm -f "${LOCK_FILE}"
            echo "CRASH: Experiment crashed with no RESULT line. State restored from best."
            exit 0
        fi

        # Extract RESULT json payload
        result_json_str="${result_line#RESULT }"
        echo "${result_json_str}" > "${exp_dir}/result.json"

        # Evaluate decision via helper
        decision_raw=$(python lab/runner_helper.py decision --result "${exp_dir}/result.json" --best "${BEST_METRIC}")
        status=$(python -c "import json; print(json.loads('''${decision_raw}''')['status'])")
        metric_val=$(python -c "import json; print(json.loads('''${decision_raw}''')['metric'])")
        metric_name=$(python -c "import json; print(json.loads('''${decision_raw}''')['metric_name'])")
        std_val=$(python -c "import json; print(json.loads('''${decision_raw}''')['std'])")
        valid_val=$(python -c "import json; print(json.loads('''${decision_raw}''')['valid'])")
        sec_val=$(python -c "import json; print(json.loads('''${decision_raw}''')['secondary'])")
        constraints_val=$(python -c "import json; print(json.loads('''${decision_raw}''')['constraints'])")
        reason_val=$(python -c "import json; print(json.loads('''${decision_raw}''').get('reason', ''))")

        # Handle keep vs discard/invalid
        if [ "${status}" = "keep" ]; then
            # Snapshot new best
            rsync -a --delete src/ "${BEST_SRC}/"
            cat <<EOF > "${BEST_METRIC}"
{"id": "${exp_id}", "metric": ${metric_val}, "metric_name": "${metric_name}", "std": ${std_val}}
EOF
            echo "KEEP: ${reason_val:-Baseline kept} (${metric_name}=${metric_val})"
        else
            # Discard or Invalid: restore src/ from lab/best/src/
            if [ -d "${BEST_SRC}" ] && [ "$(ls -A "${BEST_SRC}")" ]; then
                rsync -a --delete "${BEST_SRC}/" src/
            fi
            echo "${status^^}: ${reason_val}"
        fi

        # Append to ledger
        python lab/runner_helper.py append_ledger \
            --id "${exp_id}" --tag "${tag}" --status "${status}" --metric "${metric_val}" \
            --metric-name "${metric_name}" --std "${std_val}" --valid "${valid_val}" \
            --secondary "${sec_val}" --parent "${parent_id}" --diff "${diff_summary}" \
            --constraints "${constraints_val}" --desc "${desc}"

        # Remove lock
        rm -f "${LOCK_FILE}"
        ;;

    brief)
        python lab/runner_helper.py brief
        ;;

    show)
        target_id="${2:-}"
        if [ -z "${target_id}" ]; then
            echo "Usage: $0 show <id>" >&2
            exit 1
        fi
        
        row=$(awk -F'\t' -v tid="${target_id}" '$1 == tid {print $0}' "${LEDGER_FILE}" || true)
        if [ -n "${row}" ]; then
            echo "=== Ledger Entry: ${target_id} ==="
            echo "${row}"
        else
            echo "Experiment ${target_id} not found in ledger."
        fi

        patch_file="lab/exp/${target_id}/diff.patch"
        if [ -f "${patch_file}" ]; then
            echo ""
            echo "=== diff.patch (first 40 lines) ==="
            head -n 40 "${patch_file}"
        fi
        ;;

    *)
        echo "Usage: $0 {start <tag> \"<desc>\" | wait | status | brief | show <id>}" >&2
        exit 1
        ;;
esac

