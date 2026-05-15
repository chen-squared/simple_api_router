#!/usr/bin/env bash
# service.sh — manage simple-api-router as a macOS launchd service
#
# Usage:
#   ./scripts/service.sh install   — generate plist and load service (auto-start on login)
#   ./scripts/service.sh uninstall — unload and remove plist
#   ./scripts/service.sh start     — start service
#   ./scripts/service.sh stop      — stop service
#   ./scripts/service.sh restart   — restart service
#   ./scripts/service.sh status    — print service state
#   ./scripts/service.sh log       — tail stdout/stderr logs
#
# Hot reload: just save config.yaml — the router picks up changes automatically.
# No restart needed for provider/model/key changes.
# Host/port changes DO require a restart.

set -euo pipefail

SERVICE_LABEL="com.chen-squared.simple-api-router"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_LABEL}.plist"
LOG_DIR="$HOME/Library/Logs/simple-api-router"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
_info()  { printf '  \033[34m→\033[0m %s\n' "$*"; }
_ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
_warn()  { printf '  \033[33m!\033[0m %s\n' "$*" >&2; }
_die()   { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

_find_executable() {
    # Prefer the active venv / PATH executable
    if command -v simple-api-router &>/dev/null; then
        command -v simple-api-router
    else
        _die "simple-api-router not found in PATH. Install it first: pip install -e ."
    fi
}

_service_loaded() {
    launchctl list 2>/dev/null | grep -q "$SERVICE_LABEL"
}

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
cmd_install() {
    _bold "Installing simple-api-router service"

    local exe config_path work_dir

    exe=$(_find_executable)
    _ok "Executable: $exe"

    # Resolve config path
    if [[ -f "config.yaml" ]]; then
        config_path="$(pwd)/config.yaml"
    else
        read -r -p "  Path to config.yaml: " config_path
        config_path="${config_path/#\~/$HOME}"
    fi
    [[ -f "$config_path" ]] || _die "Config file not found: $config_path"
    _ok "Config: $config_path"

    work_dir="$(dirname "$config_path")"

    # Create log directory
    mkdir -p "$LOG_DIR"
    _ok "Log dir: $LOG_DIR"

    # Env file hint
    local env_file="$HOME/.config/simple-api-router/env"
    if [[ ! -f "$env_file" ]]; then
        mkdir -p "$(dirname "$env_file")"
        cat >"$env_file" <<'EOF'
# Environment variables for simple-api-router
# Uncomment and fill in your API keys:
#
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# DEEPSEEK_API_KEY=sk-...
EOF
        _info "Created env file template: $env_file"
        _warn "Edit $env_file and add your API keys before starting"
    fi

    # Generate wrapper script that sources the env file
    local wrapper="$HOME/.config/simple-api-router/run.sh"
    cat >"$wrapper" <<EOF
#!/bin/bash
ENV_FILE="\$HOME/.config/simple-api-router/env"
if [ -f "\$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "\$ENV_FILE"
    set +a
fi
exec "$exe" --config "$config_path"
EOF
    chmod +x "$wrapper"
    _ok "Wrapper: $wrapper"

    # Write plist
    cat >"$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${SERVICE_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${wrapper}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${work_dir}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/stderr.log</string>

    <key>ThrottleInterval</key>
    <integer>5</integer>
</dict>
</plist>
EOF
    _ok "Plist: $PLIST_PATH"

    # Load
    if _service_loaded; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
    fi
    launchctl load -w "$PLIST_PATH"
    _ok "Service loaded and enabled"
    echo
    _bold "Service installed. Useful commands:"
    echo "  ./scripts/service.sh status"
    echo "  ./scripts/service.sh log"
    echo "  ./scripts/service.sh stop / start / restart"
    echo
    _info "Hot reload: just save config.yaml — changes apply automatically."
    _info "To change API keys, edit: $env_file  then restart the service."
}

# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------
cmd_uninstall() {
    _bold "Uninstalling simple-api-router service"
    if _service_loaded; then
        launchctl unload -w "$PLIST_PATH" 2>/dev/null || true
        _ok "Service unloaded"
    fi
    if [[ -f "$PLIST_PATH" ]]; then
        rm "$PLIST_PATH"
        _ok "Plist removed"
    fi
    _ok "Done (logs remain in $LOG_DIR)"
}

# ---------------------------------------------------------------------------
# start / stop / restart
# ---------------------------------------------------------------------------
cmd_start() {
    [[ -f "$PLIST_PATH" ]] || _die "Service not installed. Run: ./scripts/service.sh install"
    launchctl start "$SERVICE_LABEL"
    _ok "Started"
}

cmd_stop() {
    launchctl stop "$SERVICE_LABEL" 2>/dev/null && _ok "Stopped" || _warn "Service was not running"
}

cmd_restart() {
    cmd_stop || true
    sleep 1
    cmd_start
}

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
cmd_status() {
    if ! _service_loaded; then
        _warn "Service not loaded (not installed or unloaded)"
        return
    fi
    echo
    launchctl print "gui/$UID/$SERVICE_LABEL" 2>/dev/null || launchctl list "$SERVICE_LABEL"
    echo
    if [[ -f "$LOG_DIR/stdout.log" ]]; then
        _info "Last 5 log lines:"
        tail -5 "$LOG_DIR/stdout.log" | sed 's/^/    /'
    fi
}

# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------
cmd_log() {
    local out="$LOG_DIR/stdout.log"
    local err="$LOG_DIR/stderr.log"
    [[ -f "$out" ]] || _die "No log found at $out — is the service installed?"
    _info "Tailing logs (Ctrl+C to stop)…"
    # Merge stdout + stderr, sorted by timestamp
    tail -f "$out" "$err"
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
case "${1:-help}" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    log)       cmd_log ;;
    *)
        echo "Usage: $0 {install|uninstall|start|stop|restart|status|log}"
        exit 1
        ;;
esac
