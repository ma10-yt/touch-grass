#!/bin/sh
set -eu

ollama serve &
server_pid=$!

stop_server() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap stop_server INT TERM

attempt=0
until ollama list >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "Ollama did not become ready within 120 seconds." >&2
        exit 1
    fi
    sleep 2
done

ollama pull gemma3:1b
ollama show gemma3:1b >/dev/null
echo "Ollama is ready and gemma3:1b is available."

wait "$server_pid"
