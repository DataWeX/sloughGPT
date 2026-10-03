#!/usr/bin/env bash
# backup_critical.sh — extinction-grade backup: journal, boards, code.
#
# Legs:
#   local  : dated slog-backup-<ts>.tgz (journal + both boards, optional full
#            git bundle) + sha256 manifest, retention, --verify re-hash
#   offsite: git push --all --tags (code repo) + journal repo commit/push
#            to its private GitHub origin (board snapshots refreshed inside)
#
# usage:
#   backup_critical.sh                 # full run: local tar + push legs
#   backup_critical.sh --local         # local tar only (offline / no push)
#   backup_critical.sh --full          # add a git bundle of the code store
#   backup_critical.sh --verify [file] # re-hash one archive (default: all)
#
# env: SLOG_JOURNAL, SLOG_BACKUP_DIR, SLOG_CODE_ROOT, SLOG_BACKUP_KEEP
set -euo pipefail

JOURNAL="${SLOG_JOURNAL:-$HOME/.config/dev-notes}"
BACKUP_DIR="${SLOG_BACKUP_DIR:-$HOME/.cache/slog-backup}"
CODE_ROOT="${SLOG_CODE_ROOT:-/home/mana/Documents/Default Project/sloughGPT}"
API_STD="${SLOG_API_STD:-/home/mana/Documents/sloughgpt-api-std}"
RETENTION="${SLOG_BACKUP_KEEP:-10}"

ts() { date -u +%Y%m%dT%H%M%SZ; }

log() { printf '[backup] %s\n' "$*"; }
warn() { printf '[backup] WARN: %s\n' "$*" >&2; }

verify_archives() {
    local target="${1:-}" rc=0 shas
    if [[ -n "$target" && "$target" != "all" ]]; then
        shas=("$target")
    else
        mapfile -t shas < <(ls -1t "$BACKUP_DIR"/slog-backup-*.tgz.sha256 2>/dev/null || true)
    fi
    if [[ ${#shas[@]} -eq 0 ]]; then
        warn "no archives to verify in $BACKUP_DIR"
        return 1
    fi
    for m in "${shas[@]}"; do
        if (cd "$(dirname "$m")" && sha256sum --check "$(basename "$m")" >/dev/null); then
            log "verify OK  $m"
        else
            warn "verify FAIL $m"
            rc=1
        fi
    done
    return $rc
}

make_tar() {
    local full="$1" stamp out stage
    stamp="$(ts)"
    out="$BACKUP_DIR/slog-backup-$stamp.tgz"
    stage="$BACKUP_DIR/.stage-$$"
    mkdir -p "$BACKUP_DIR" "$stage"
    trap 'rm -rf "$stage"' RETURN

    cp -a "$JOURNAL" "$stage/journal"
    cp -a "$API_STD/.kanban/board.jsonl" "$stage/board-api-std.jsonl"
    if [[ -f "$CODE_ROOT/.kanban/board.jsonl" ]]; then
        cp -a "$CODE_ROOT/.kanban/board.jsonl" "$stage/board-worktree.jsonl"
    fi
    if [[ "$full" == 1 ]]; then
        log "creating git bundle (all refs)..."
        git -C "$CODE_ROOT" bundle create "$stage/code.bundle" --all >/dev/null
    fi

    tar -czf "$out" -C "$stage" .
    sha256sum "$out" > "$out.sha256"
    log "wrote $out ($(du -h "$out" | cut -f1)) + manifest"

    # retention: keep newest $RETENTION archives (manifests follow their tgz)
    local i=0 f
    while read -r f; do
        i=$((i + 1))
        if (( i > RETENTION )); then
            rm -f "$f" "$f.sha256"
            log "pruned old archive $(basename "$f")"
        fi
    done < <(ls -1t "$BACKUP_DIR"/slog-backup-*.tgz 2>/dev/null)
}

push_code() {
    local rc=0
    log "pushing code branches to origin..."
    git -C "$CODE_ROOT" push origin --all --quiet || { warn "push --all failed"; rc=1; }
    git -C "$CODE_ROOT" push origin --tags --quiet || { warn "push --tags failed"; rc=1; }
    return $rc
}

push_journal() {
    local rc=0 snap="$JOURNAL/snapshots"
    mkdir -p "$snap"
    cp -a "$API_STD/.kanban/board.jsonl" "$snap/board-api-std.jsonl"
    if [[ -f "$CODE_ROOT/.kanban/board.jsonl" ]]; then
        cp -a "$CODE_ROOT/.kanban/board.jsonl" "$snap/board-worktree.jsonl"
    fi
    git -C "$JOURNAL" add -A
    if git -C "$JOURNAL" diff --cached --quiet; then
        log "journal: no changes to commit"
    else
        git -C "$JOURNAL" commit -qm "backup $(ts)" || { warn "journal commit failed"; rc=1; }
    fi
    git -C "$JOURNAL" push -u origin HEAD --quiet || { warn "journal push failed"; rc=1; }
    return $rc
}

main() {
    local mode="${1:-full}" rc=0
    case "$mode" in
        --verify)
            verify_archives "${2:-}"
            exit $?
            ;;
        --local)
            make_tar "${SLOG_BUNDLE:-0}"
            verify_archives
            exit $?
            ;;
        --full)
            make_tar 1
            verify_archives
            exit $?
            ;;
        *)
            make_tar "${SLOG_BUNDLE:-0}"
            verify_archives || rc=1
            push_code || rc=1
            push_journal || rc=1
            exit $rc
            ;;
    esac
}

main "$@"
