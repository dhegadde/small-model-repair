#!/bin/bash
# Mac resource guard: watches RAM, swap and disk while a long job runs, and STOPS the job before the
# machine gets unhealthy. Run it in the background next to any long Mac job:
#
#   bash scripts/mac_guard.sh "<pattern matching the job's command line>" [log_file]
#
# Every 20 s it logs: job memory, system free-memory %, swap used, disk free. It kills the job (SIGTERM)
# and exits with a reason if any limit is crossed; it exits quietly when the job finishes.
# Limits (defaults, override with env vars):
#   GUARD_MIN_FREE_PCT=15    system-wide free memory % (macOS memory_pressure) must stay above this
#   GUARD_MAX_SWAP_GROWTH_MB=1500   swap may not grow more than this since the guard started
#   GUARD_MAX_JOB_MB=9000    the job's own memory footprint (a 16 GB Mac needs room for macOS + apps)
#   GUARD_MIN_DISK_GB=20     free disk space on the home volume
set -u
PATTERN="$1"
LOG="${2:-/tmp/mac_guard.log}"
MIN_FREE=${GUARD_MIN_FREE_PCT:-15}
MAX_SWAP_GROWTH=${GUARD_MAX_SWAP_GROWTH_MB:-1500}
MAX_JOB=${GUARD_MAX_JOB_MB:-9000}
MIN_DISK=${GUARD_MIN_DISK_GB:-20}

swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print int($2)}'; }
disk_gb() { df -g "$HOME" | awk 'NR==2 {print $4}'; }
job_pid() { pgrep -f "$PATTERN" | while read p; do ps -o comm= -p "$p" | grep -qi python && echo "$p"; done | head -1; }
job_mb() { top -l 1 -pid "$1" -stats mem 2>/dev/null | tail -1 | awk '{v=$1; u=substr(v,length(v)); n=v+0; if(u=="G") n*=1024; else if(u=="K") n/=1024; print int(n)}'; }

SWAP0=$(swap_mb)
echo "$(date '+%H:%M:%S') guard start: pattern='$PATTERN' swap0=${SWAP0}MB limits free>=${MIN_FREE}% swap+<=${MAX_SWAP_GROWTH}MB job<=${MAX_JOB}MB disk>=${MIN_DISK}GB" >> "$LOG"
sleep 5
while true; do
  PID=$(job_pid)
  if [ -z "$PID" ]; then
    # Multi-step drivers start one python after another; only exit after 3 empty checks (~40 s).
    MISSES=$((${MISSES:-0}+1))
    if [ "$MISSES" -ge 3 ]; then echo "$(date '+%H:%M:%S') job finished; guard exiting (healthy)" >> "$LOG"; echo "GUARD: job finished, machine stayed healthy"; exit 0; fi
    sleep 20; continue
  fi
  MISSES=0
  FREE=$(free_pct); SWAP=$(swap_mb); DISK=$(disk_gb); JOB=$(job_mb "$PID")
  echo "$(date '+%H:%M:%S') pid=$PID job=${JOB}MB free=${FREE}% swap=${SWAP}MB (+$((SWAP-SWAP0))) disk=${DISK}GB" >> "$LOG"
  REASON=""
  [ "${FREE:-100}" -lt "$MIN_FREE" ] && REASON="system free memory ${FREE}% < ${MIN_FREE}%"
  [ $((SWAP-SWAP0)) -gt "$MAX_SWAP_GROWTH" ] && REASON="swap grew $((SWAP-SWAP0))MB > ${MAX_SWAP_GROWTH}MB"
  [ "${JOB:-0}" -gt "$MAX_JOB" ] && REASON="job memory ${JOB}MB > ${MAX_JOB}MB"
  [ "${DISK:-999}" -lt "$MIN_DISK" ] && REASON="disk free ${DISK}GB < ${MIN_DISK}GB"
  if [ -n "$REASON" ]; then
    kill "$PID" 2>/dev/null
    echo "$(date '+%H:%M:%S') STOPPED job $PID: $REASON" >> "$LOG"
    echo "GUARD STOPPED THE JOB: $REASON (see $LOG)"
    exit 2
  fi
  sleep 20
done
