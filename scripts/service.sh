#!/usr/bin/env bash
# service.sh — manage simple-api-router as a background service
#
# Supports:
#   macOS  — launchd  (~/Library/LaunchAgents/)
#   Linux  — systemd user service (~/.config/systemd/user/)
#
# Usage:
#   ./scripts/service.sh install [--config PATH] [--exe PATH]
#       Install and enable the service (auto-start on login).
#       Default config: ~/.config/simple-api-router/config.yaml
#   ./scripts/service.sh uninstall  — stop and remove the service
#   ./scripts/service.sh start      — start the service
#   ./scripts/service.sh stop       — stop the service
#   ./scripts/service.sh restart    — restart the service
#   ./scripts/service.sh status     — show service state
#   ./scripts/service.sh log        — tail live logs
#
# Hot reload: just save config.yaml — provider/model/key/retry changes apply
# automatically within ~1 second with no restart and no dropped connections.
# Changes to host/port require a restart.

set -euo pipefail

SERVICE_NAME="simple-api-router"
SERVICE_LABEL="com.chen-squared.simple-api-router"   # macOS
CONFIG_DIR="$HOME/.config/simple-api-router"
DEFAULT_CONFIG="$CONFIG_DIR/config.yaml"
ENV_FILE="$CONFIG_DIR/env"
WRAPPER="$CONFIG_DIR/run.sh"

# macOS paths
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_LABEL}.plist"
LOG_DIR="$HOME/Library/Logs/simple-api-router"

# Linux paths
SYSTEMD_UNIT_DIR="$HOME/.config/systemd/user"
SYSTEMD_UNIT="$SYSTEMD_UNIT_DIR/${SERVICE_NAME}.service"

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)  echo "linux" ;;
        *)      echo "unknown" ;;
    esac
}

OS="$(_os)"
[[ "$OS" == "unknown" ]] && { echo "Unsupported OS: $(uname -s)" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
_info()  { printf '  \033[34m→\033[0m %s\n' "$*"; }
_ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
_warn()  { printf '  \033[33m!\033[0m %s\n' "$*" >&2; }
_die()   { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

_find_executable() {
    # 1. Explicit --exe argument
    if [[ -n "${_EXE_OVERRIDE:-}" ]]; then
        [[ -x "$_EXE_OVERRIDE" ]] || _die "Not executable: $_EXE_OVERRIDE"
        echo "$_EXE_OVERRIDE"; return
    fi
    # 2. Already in PATH (venv active or system install)
    if command -v simple-api-router &>/dev/null; then
        command -v simple-api-router; return
    fi
    # 3. Common venv / install locations
    local candidates=(
        "$HOME/Developer/.venv/bin/simple-api-router"
        "$HOME/.venv/bin/simple-api-router"
        "$HOME/venv/bin/simple-api-router"
        "$HOME/.local/bin/simple-api-router"
        "/usr/local/bin/simple-api-router"
    )
    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]]; then echo "$c"; return; fi
    done
    _die "simple-api-router not found. Either:
  • activate your venv before running install, or
  • pass the path: $0 install --exe /path/to/.venv/bin/simple-api-router"
}

_resolve_config() {
    local explicit="$1"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [[ -n "$explicit" ]]; then
        echo "${explicit/#\~/$HOME}"; return
    fi
    if [[ -f "$DEFAULT_CONFIG" ]]; then
        echo "$DEFAULT_CONFIG"; return
    fi
    if [[ -f "$(pwd)/config.yaml" ]]; then
        echo "$(pwd)/config.yaml"; return
    fi
    mkdir -p "$CONFIG_DIR"
    if [[ -f "$script_dir/../config.yaml" ]]; then
        cp "$script_dir/../config.yaml" "$DEFAULT_CONFIG"
        _info "Copied config template to $DEFAULT_CONFIG"
    else
        touch "$DEFAULT_CONFIG"
        _info "Created empty config at $DEFAULT_CONFIG"
    fi
    echo "$DEFAULT_CONFIG"
}

_ensure_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        mkdir -p "$CONFIG_DIR"
        cat >"$ENV_FILE" <<'EOF'
# Environment variables for simple-api-router
# Uncomment and fill in your API keys:
#
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# DEEPSEEK_API_KEY=sk-...
EOF
        _info "Created env file template: $ENV_FILE"
        _warn "Edit $ENV_FILE and add your API keys before starting"
    fi
}

_write_wrapper() {
    local exe="$1" config_path="$2"
    cat >"$WRAPPER" <<EOF
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
    chmod +x "$WRAPPER"
    _ok "Wrapper: $WRAPPER"
}

# ---------------------------------------------------------------------------
# macOS — launchd
# ---------------------------------------------------------------------------
_launchd_loaded() {
    launchctl list "$SERVICE_LABEL" &>/dev/null
}

_install_macos() {
    local exe="$1" config_path="$2" work_dir="$3"
    mkdir -p "$LOG_DIR"
    _ok "Log dir: $LOG_DIR"
    _write_wrapper "$exe" "$config_path"
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
        <string>${WRAPPER}</string>
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
    if _launchd_loaded; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
    fi
    launchctl load -w "$PLIST_PATH"
    _ok "Service loaded and enabled (auto-starts on login)"
}

_uninstall_macos() {
    if _launchd_loaded; then
        launchctl unload -w "$PLIST_PATH" 2>/dev/null || true
        _ok "Service unloaded"
    fi
    [[ -f "$PLIST_PATH" ]] && rm "$PLIST_PATH" && _ok "Plist removed"
}

_start_macos()   { [[ -f "$PLIST_PATH" ]] || _die "Not installed. Run: $0 install"; launchctl start "$SERVICE_LABEL"; _ok "Started"; }
_stop_macos()    { launchctl stop "$SERVICE_LABEL" 2>/dev/null && _ok "Stopped" || _warn "Was not running"; }
_restart_macos() { _stop_macos || true; sleep 1; _start_macos; }

_status_macos() {
    if ! _launchd_loaded; then _warn "Service not loaded"; return; fi
    launchctl print "gui/$UID/$SERVICE_LABEL" 2>/dev/null || launchctl list "$SERVICE_LABEL"
    if [[ -f "$LOG_DIR/stdout.log" ]]; then
        echo; _info "Last 5 log lines:"; tail -5 "$LOG_DIR/stdout.log" | sed 's/^/    /'
    fi
}

_log_macos() {
    [[ -f "$LOG_DIR/stdout.log" ]] || _die "No logs at $LOG_DIR — is the service installed?"
    _info "Tailing logs (Ctrl+C to stop)…"
    tail -f "$LOG_DIR/stdout.log" "$LOG_DIR/stderr.log"
}

# ---------------------------------------------------------------------------
# Linux — systemd user
# ---------------------------------------------------------------------------
_systemd() { systemctl --user "$@"; }

_install_linux() {
    local exe="$1" config_path="$2" work_dir="$3"
    _write_wrapper "$exe" "$config_path"
    mkdir -p "$SYSTEMD_UNIT_DIR"
    cat >"$SYSTEMD_UNIT" <<EOF
[Unit]
Description=Simple API Router
After=network.target

[Service]
Type=simple
ExecStart=${WRAPPER}
WorkingDirectory=${work_dir}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
    _ok "Unit file: $SYSTEMD_UNIT"
    _systemd daemon-reload
    _systemd enable "$SERVICE_NAME"
    _systemd start  "$SERVICE_NAME"
    _ok "Service enabled and started"
    if command -v loginctl &>/dev/null; then
        loginctl enable-linger "$USER" 2>/dev/null && _ok "Linger enabled (service survives logout)"
    fi
}

_uninstall_linux() {
    if _systemd is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        _systemd stop "$SERVICE_NAME"; _ok "Stopped"
    fi
    if _systemd is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        _systemd disable "$SERVICE_NAME"; _ok "Disabled"
    fi
    [[ -f "$SYSTEMD_UNIT" ]] && rm "$SYSTEMD_UNIT" && _systemd daemon-reload && _ok "Unit file removed"
}

_start_linux()   { _systemd start   "$SERVICE_NAME"; _ok "Started"; }
_stop_linux()    { _systemd stop    "$SERVICE_NAME"; _ok "Stopped"; }
_restart_linux() { _systemd restart "$SERVICE_NAME"; _ok "Restarted"; }
_status_linux()  { _systemd status  "$SERVICE_NAME" --no-pager || true; }
_log_linux()     { _info "Tailing logs (Ctrl+C to stop)…"; journalctl --user -u "$SERVICE_NAME" -f --no-pager; }

# ---------------------------------------------------------------------------
# Cross-platform dispatch
# ---------------------------------------------------------------------------
_install() {
    local explicit_config="" _EXE_OVERRIDE=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config|-c) shift; explicit_config="$1" ;;
            --exe|-e)    shift; _EXE_OVERRIDE="${1/#\~/$HOME}" ;;
            *) _die "Unknown argument: $1" ;;
        esac
        shift
    done

    _bold "Installing $SERVICE_NAME service ($OS)"

    local exe config_path work_dir
    exe=$(_find_executable);  _ok "Executable: $exe"
    config_path=$(_resolve_config "$explicit_config")
    config_path="$(cd "$(dirname "$config_path")" && pwd)/$(basename "$config_path")"
    [[ -f "$config_path" ]] || _die "Config not found: $config_path"
    _ok "Config: $config_path"
    work_dir="$(dirname "$config_path")"

    _ensure_env_file

    case "$OS" in
        macos) _install_macos "$exe" "$config_path" "$work_dir" ;;
        linux) _install_linux "$exe" "$config_path" "$work_dir" ;;
    esac

    echo
    _bold "Done. Useful commands:"
    echo "  $0 status / log / stop / start / restart / uninstall"
    _info "Hot reload: just save config.yaml — changes apply automatically."
    _info "API keys:   edit $ENV_FILE  then restart."
}

_uninstall() {
    _bold "Uninstalling $SERVICE_NAME service"
    case "$OS" in
        macos) _uninstall_macos ;;
        linux) _uninstall_linux ;;
    esac
    _ok "Done (logs and env file preserved)"
}

_start()   { case "$OS" in macos) _start_macos   ;; linux) _start_linux   ;; esac; }
_stop()    { case "$OS" in macos) _stop_macos    ;; linux) _stop_linux    ;; esac; }
_restart() { case "$OS" in macos) _restart_macos ;; linux) _restart_linux ;; esac; }
_status()  { case "$OS" in macos) _status_macos  ;; linux) _status_linux  ;; esac; }
_log()     { case "$OS" in macos) _log_macos     ;; linux) _log_linux     ;; esac; }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
case "${1:-help}" in
    install)   shift; _install "$@" ;;
    uninstall) _uninstall ;;
    start)     _start ;;
    stop)      _stop ;;
    restart)   _restart ;;
    status)    _status ;;
    log)       _log ;;
    *)
        echo "Usage: $0 {install [--config PATH] [--exe PATH]|uninstall|start|stop|restart|status|log}"
        echo "       Supports macOS (launchd) and Linux (systemd user service)"
        exit 1
        ;;
esac
