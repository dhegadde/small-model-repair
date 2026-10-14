#!/bin/bash
# Sync a pod's run directory to the Mac and STOP THE POD automatically when:
#   - the driver log says "ALL DONE" or "FAILED";
#   - the pod is unreachable for ~15 min (5 missed checks);
#   - NOTHING in the run dir or driver log has been written for STALL_MIN minutes (default 45): a silently hung
#     job (2026-10-01: a full network volume froze every later step; the pod billed ~6.5 idle hours, ~$4.80);
#   - the network volume (/workspace) has less than MIN_FREE_GB free (default 2): a full disk hangs jobs silently.
# Never rely on a person or agent to notice: an idle pod after a failed run cost ~$6 on 2026-09-25.
#   caffeinate -i bash scripts/pod_watch.sh <pod_id> <ip> <port> <remote_run_dir> <remote_driver_log> <local_run_dir>
#   (caffeinate: a sleeping Mac paused the watcher for 80 min on 2026-09-25)
set -u
POD=$1; IP=$2; PORT=$3; RDIR=$4; RLOG=$5; LDIR=$6
STALL_MIN=${STALL_MIN:-45}; MIN_FREE_GB=${MIN_FREE_GB:-2}; VOLUME_GB=${VOLUME_GB:-30}  # quota; df shows the whole shared FS
# keepalives: a stalled connection is dropped after ~60 s instead of hanging the watcher forever (2026-10-09: an
# rsync hung after the job finished, the watcher never saw ALL DONE, and the pod idled ~1.75 h, ~$1.60)
K=$HOME/.runpod/ssh/runpodctl-ssh-key; SSH="ssh -i $K -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p $PORT"
START=$(date +%s); mkdir -p "$LDIR"; LLOG="$LDIR/$(basename "$RLOG")"; misses=0

stop_pod() {  # $1 = reason
  echo "STOPPING ($1) $(date -u +%H:%M:%S)"
  runpodctl pod stop "$POD" >/dev/null 2>&1 && echo "POD $POD STOPPED $(date -u +%H:%M:%S)" \
    || echo "POD STOP FAILED for $POD: check it now with list-pods / runpodctl pod list"
  tail -5 "$LLOG" 2>/dev/null
  exit 0
}

while true; do
  # the driver log first (small): the stop decision must never wait behind a big results sync
  if rsync -az --timeout=60 -e "$SSH" "root@$IP:$RLOG" "$LLOG" 2>/dev/null; then misses=0; else misses=$((misses+1)); fi
  grep -qE "ALL DONE|FAILED" "$LLOG" 2>/dev/null && { rsync -az --timeout=300 -e "$SSH" --exclude='adapter/' --exclude='*.pt' \
    --exclude='*.safetensors' "root@$IP:$RDIR/" "$LDIR/" 2>/dev/null; stop_pod "driver finished"; }
  rsync -az --timeout=120 -e "$SSH" --exclude='adapter/' --exclude='*.pt' --exclude='*.safetensors' "root@$IP:$RDIR/" "$LDIR/" 2>/dev/null
  [ $misses -ge 5 ] && stop_pod "pod unreachable"
  if [ $misses -eq 0 ]; then
    # one remote check: files written in the last STALL_MIN minutes, and free GB on the volume
    read -r recent free_gb <<<"$($SSH root@$IP "n=\$(find '$RDIR' '$RLOG' -newermt '-$STALL_MIN minutes' 2>/dev/null | head -1 | wc -l); \
      u=\$(du -s -BG /workspace 2>/dev/null | tr -dc 0-9); echo \$n \$(( $VOLUME_GB - u ))" 2>/dev/null)"
    if [ -n "${recent:-}" ] && [ "$recent" = 0 ] && [ "$(( $(date +%s) - START ))" -gt $(( STALL_MIN * 60 )) ]; then
      stop_pod "no file written for $STALL_MIN min: job looks hung"
    fi
    if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
      stop_pod "network volume nearly full (${free_gb} GB free)"
    fi
  fi
  sleep 180
done
