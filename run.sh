#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SERVER_9999_LOG="${LOG_DIR}/server_9999.log"
SERVER_9998_LOG="${LOG_DIR}/server_9998.log"

: > "${SERVER_9999_LOG}"
: > "${SERVER_9998_LOG}"

SERVER_PIDS=()

cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local port="$1"
  local log_file="$2"
  local pid="$3"
  local pattern="listening on .*:${port}"
  local timeout_sec=30
  local start
  start="$(date +%s)"

  while true; do
    if grep -E "${pattern}" "${log_file}" >/dev/null 2>&1; then
      echo "server ${port} is ready"
      return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "server ${port} exited before it became ready. Log:"
      tail -n 80 "${log_file}" || true
      return 1
    fi

    if (( "$(date +%s)" - start >= timeout_sec )); then
      echo "timed out waiting for server ${port}. Log:"
      tail -n 80 "${log_file}" || true
      return 1
    fi

    sleep 0.25
  done
}

cd "${ROOT_DIR}"

echo "starting server on port 9999"
python -u -m src.inference.server_h264 --port 9999 --show-window > "${SERVER_9999_LOG}" 2>&1 &
SERVER_PIDS+=("$!")

echo "starting server on port 9998"
python -u -m src.inference.server_h264 --port 9998 --show-window > "${SERVER_9998_LOG}" 2>&1 &
SERVER_PIDS+=("$!")

wait_for_server 9999 "${SERVER_9999_LOG}" "$!"
wait_for_server 9998 "${SERVER_9998_LOG}" "$!"

echo "both servers are ready; starting camera"
python -m src.streaming.camera_h264
