#!/usr/bin/env bash
set -euo pipefail

# Autoresearch Robotics Outer Loop
# Spawns fresh, isolated agent sessions for each block of experiments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Autoresearch Robotics Outer Loop"
    echo "Usage: ./lab/loop.sh"
    echo ""
    echo "Runs continuous agent sessions until 'lab/STOP' is created."
    echo "Every third session is 'exploit'; other sessions are 'explore'."
    echo ""
    echo "Environment Variables:"
    echo "  AGENT_CMD          Default command to invoke your coding agent"
    echo "  AGENT_IDEATOR      Command for explore sessions (defaults to AGENT_CMD)"
    echo "  AGENT_CMD_EXPLOIT  Command for exploit sessions (defaults to AGENT_CMD)"
    echo "  MAX_EXPERIMENTS    Maximum experiments per session (default: 5)"
    echo ""
    echo "To stop the loop safely:"
    echo "  touch lab/STOP"
    exit 0
fi

MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-5}"
DEFAULT_AGENT_CMD=""

if [ -z "${AGENT_CMD:-}" ] && [ -z "${AGENT_IDEATOR:-}" ]; then
    echo "Error: No agent command configured." >&2
    echo "Please set AGENT_CMD or AGENT_IDEATOR environment variable." >&2
    echo "Run '$0 --help' for details." >&2
    exit 1
fi

AGENT_CMD="${AGENT_CMD:-${AGENT_IDEATOR:-}}"
AGENT_IDEATOR="${AGENT_IDEATOR:-${AGENT_CMD}}"
AGENT_CMD_EXPLOIT="${AGENT_CMD_EXPLOIT:-${AGENT_CMD}}"

LOOP_LOG="lab/loop.log"
STOP_FILE="lab/STOP"

# Remove any old stop file before starting
rm -f "${STOP_FILE}"

echo "==========================================================" | tee -a "${LOOP_LOG}"
echo "[$(date -Iseconds)] Starting Autoresearch Robotics Loop" | tee -a "${LOOP_LOG}"
echo "==========================================================" | tee -a "${LOOP_LOG}"

session_count=1

while true; do
    if [ -f "${STOP_FILE}" ]; then
        echo "[$(date -Iseconds)] STOP file detected (${STOP_FILE}). Exiting loop." | tee -a "${LOOP_LOG}"
        rm -f "${STOP_FILE}"
        break
    fi

    # Determine mode: every 3rd session is exploit; others are explore
    if [ $((session_count % 3)) -eq 0 ]; then
        MODE="exploit"
        CURRENT_AGENT_CMD="${AGENT_CMD_EXPLOIT}"
    else
        MODE="explore"
        CURRENT_AGENT_CMD="${AGENT_IDEATOR}"
    fi

    session_id=$(printf "S%03d" "${session_count}")
    launch_msg="SESSION_ID=${session_id} MODE=${MODE} MAX_EXPERIMENTS=${MAX_EXPERIMENTS}"

    echo "[$(date -Iseconds)] Launching Session ${session_id} (Mode: ${MODE})" | tee -a "${LOOP_LOG}"

    # Execute agent command with launch message
    set +e
    eval "${CURRENT_AGENT_CMD} \"${launch_msg}\"" >> "${LOOP_LOG}" 2>&1
    exit_code=$?
    set -e

    echo "[$(date -Iseconds)] Session ${session_id} finished with code ${exit_code}." | tee -a "${LOOP_LOG}"

    # Check for stop signal after session completes
    if [ -f "${STOP_FILE}" ]; then
        echo "[$(date -Iseconds)] STOP file detected (${STOP_FILE}). Exiting loop." | tee -a "${LOOP_LOG}"
        rm -f "${STOP_FILE}"
        break
    fi

    session_count=$((session_count + 1))
    sleep 2
done

echo "[$(date -Iseconds)] Loop terminated cleanly." | tee -a "${LOOP_LOG}"

