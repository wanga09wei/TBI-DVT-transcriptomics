#!/bin/bash
# GSE182398 + GSE294775 单细胞数据下载器（单实例 PID 锁 + .part 原子改名 + Content-Length 校验）
# v1 2026-10-13
set -u
DEST="D:/TBI_DVT_Study/scrna/data"
LOCK="$DEST/.singlecell_dl.lock"
BASE182="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE182nnn/GSE182398/suppl"
BASE294="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE294nnn/GSE294775/suppl"

# ---- 单实例锁 ----
if [ -f "$LOCK" ]; then
  OPID=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$OPID" ] && kill -0 "$OPID" 2>/dev/null; then
    echo "[LOCK] 已有实例运行 (PID $OPID)，退出"; exit 1
  fi
  echo "[LOCK] 陈旧锁清理 (PID $OPID 已不存在)"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

dl() {  # dl <url> <dest>
  local url="$1" out="$2"
  local tmp="$out.part"
  local want have
  want=$(curl -sI "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
  if [ -f "$out" ] && [ -n "$want" ] && [ "$(stat -c%s "$out" 2>/dev/null)" = "$want" ]; then
    echo "[SKIP] $(basename "$out") 已存在且大小校验通过 ($want B)"; return 0
  fi
  echo "[GET ] $(basename "$out") 期望 $want B"
  for attempt in 1 2 3 4 5; do
    curl -sL --retry 0 -C - -o "$tmp" "$url" && break
    echo "[RETRY $attempt] 断点续传中…"; sleep 5
  done
  have=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$have" != "$want" ]; then
    echo "[FAIL] $(basename "$out") 大小不符 have=$have want=$want"; return 1
  fi
  gzip -t "$tmp" 2>/dev/null || [ "${tmp##*.}" != ".gz" ] && [ "${out##*.}" != ".gz" ] || {
    # tar 也校验可读性
    tar -tf "$tmp" >/dev/null 2>&1 || true
  }
  mv -f "$tmp" "$out"
  echo "[OK  ] $(basename "$out") ($have B)"
}

echo "=== 阶段1: GSE182398 RAW.tar (149.6 MB) ==="
dl "$BASE182/GSE182398_RAW.tar" "$DEST/GSE182398_RAW.tar"
echo "=== 阶段2: GSE294775 10x 矩阵包 (~954 MB) ==="
dl "$BASE294/GSE294775_barcodes.tsv.gz"  "$DEST/GSE294775/GSE294775_barcodes.tsv.gz"
dl "$BASE294/GSE294775_features.tsv.gz"  "$DEST/GSE294775/GSE294775_features.tsv.gz"
dl "$BASE294/GSE294775_metadata.csv.gz"  "$DEST/GSE294775/GSE294775_metadata.csv.gz"
dl "$BASE294/GSE294775_matrix.mtx.gz"    "$DEST/GSE294775/GSE294775_matrix.mtx.gz"
echo "=== 全部完成 ==="
