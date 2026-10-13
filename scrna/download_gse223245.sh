#!/bin/bash
# =====================================================================
# GSE223245 逐样本下载器 v2（D 盘版）
# 改进要点（相对 v1，v1 因多实例并发写同一文件导致数据损坏）：
#   1. 单实例锁：PID 锁文件，已有存活实例则拒绝启动，杜绝并发写
#   2. .part 原子改名：先下 .part，校验大小通过后才 mv 为正式文件
#   3. 远端大小校验：以 HTTP Content-Length 为准，不用固定阈值猜
#   4. 3 并发（NCBI 限流友好）+ 断点续传，可反复中断重启
# 用法：bash download_gse223245.sh          （前台/受管后台皆可）
#       bash download_gse223245.sh --status （只查状态，不下载）
# =====================================================================
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
DATA="$DIR/data"
LOG="$DIR/download.log"
mkdir -p "$DATA"
cd "$DATA" || exit 1

SAMPLES="
GSM6943813_Mi1
GSM6943814_Mi2
GSM6943815_Mi3
GSM6943816_Mi4
GSM6943817_Mo1
GSM6943818_Mo2
GSM6943819_Mo3
GSM6943820_Mo4
GSM6943821_Se1
GSM6943822_Se2
GSM6943823_Se3
GSM6943824_Se4
GSM6943825_Co1
GSM6943826_Co2
GSM6943827_Co3
GSM6943828_Co4
"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/samples"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

remote_size(){
  curl -sI --max-time 60 "$1" | awk 'BEGIN{IGNORECASE=1} /^content-length:/{gsub(/\r/,"");print $2}' | tail -1
}

# ---------- 状态查询模式 ----------
if [ "${1:-}" = "--status" ]; then
  done_n=0; part_n=0; miss_n=0
  for s in $SAMPLES; do
    if [ -f "$s.txt.gz" ]; then
      sz=$(stat -c%s "$s.txt.gz")
      if [ "$sz" -gt 20000000 ]; then echo "  [完成] $s.txt.gz  $((sz/1048576))MB"; done_n=$((done_n+1))
      else echo "  [残缺] $s.txt.gz  $((sz/1024))KB"; part_n=$((part_n+1)); fi
    elif [ -f "$s.txt.gz.part" ]; then
      echo "  [下载中] $s.txt.gz.part  $(($(stat -c%s "$s.txt.gz.part")/1024))KB"; part_n=$((part_n+1))
    else echo "  [未开始] $s"; miss_n=$((miss_n+1)); fi
  done
  echo "---- 完成 $done_n / 未完成 $part_n / 未开始 $miss_n (共 16) ----"
  exit 0
fi

# ---------- 单实例锁 ----------
LOCK="$DATA/.download.lock"
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null)
  if [ -n "${oldpid:-}" ] && kill -0 "$oldpid" 2>/dev/null; then
    log "[ABORT] 已有下载实例在运行 (PID $oldpid)，拒绝启动以避免并发写损坏文件"
    exit 1
  fi
  log "[warn] 陈旧锁文件（PID ${oldpid:-未知} 已不存在），接管"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

dl_one(){
  local s="$1"
  local fn="$s.txt.gz"
  local part="$fn.part"
  local gsm="${s%%_*}"
  local nn="${gsm:0:7}nnn"
  local url="$BASE/$nn/$gsm/suppl/$fn"

  local rsz
  rsz=$(remote_size "$url")
  if [ -z "$rsz" ]; then log "[skip-url] $s 无法获取远端大小（稍后重试）"; return 1; fi

  # 已完整
  if [ -f "$fn" ] && [ "$(stat -c%s "$fn")" = "$rsz" ]; then
    log "[done] $fn ($((rsz/1048576))MB) 已完整，跳过"; return 0
  fi

  local attempt=0
  while [ "$attempt" -lt 40 ]; do
    attempt=$((attempt+1))
    curl -s -C - -o "$part" --max-time 1800 "$url"
    local cur
    cur=$(stat -c%s "$part" 2>/dev/null || echo 0)
    if [ "$cur" = "$rsz" ]; then
      mv -f "$part" "$fn"
      log "[done] $fn ($((rsz/1048576))MB) 第 ${attempt} 次尝试完成"
      return 0
    fi
    log "[retry] $s 第 ${attempt} 次：$((cur/1024))KB / $((rsz/1024))KB"
    sleep 3
  done
  log "[FAIL] $s 超过重试上限，剩余 $(( $(stat -c%s "$part" 2>/dev/null || echo 0)/1024))KB"
  return 1
}

log "=== 开始下载 GSE223245（16 样本，3 并发，PID $$）==="
for s in $SAMPLES; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
  dl_one "$s" &
done
wait

echo "=== 汇总 ===" | tee -a "$LOG"
for s in $SAMPLES; do
  if [ -f "$s.txt.gz" ] && [ "$(stat -c%s "$s.txt.gz")" -gt 20000000 ]; then
    echo "  [OK]   $s.txt.gz" | tee -a "$LOG"
  else
    echo "  [MISS] $s" | tee -a "$LOG"
  fi
done
log "=== ALL SAMPLES DONE ==="
ls -la *.txt.gz 2>/dev/null | tee -a "$LOG"
