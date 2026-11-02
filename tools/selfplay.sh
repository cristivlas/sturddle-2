#!/usr/bin/env bash
#
# Self-play data generation under fastchess: ENGINE vs itself (default: sturddle).
# Sturddle-specific options are passed only if the engine's filename has "sturddle".
# Optionally pinned to a single NUMA node via numactl.
#
set -euo pipefail

ROOT=$(dirname "$(dirname "$(realpath "$0")")")

FASTCHESS=${FASTCHESS:-$HOME/Projects/fastchess/fastchess}
ENGINE=${ENGINE:-$ROOT/dist/sturddle-2.8.0-Linux-x86_64}
TB=${TB:-$HOME/Projects/syzygy/3-4-5}
BOOK=${BOOK:-$ROOT/tuneup/books/megabook.epd}
HASH=${HASH:-256}
DEPTH=${DEPTH:-16}
PLIES=${PLIES:-2}
OUTDIR=${OUTDIR:-$ROOT/train-data}
LOG_DIR=${LOG_DIR:-}
GAMES_DEFAULT=1000000

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<USAGE
Usage: ${0##*/} [options] [games]

  [games]           number of games (default: $GAMES_DEFAULT)

Options:
  -N, --node <n>    pin CPU and memory to NUMA node <n> via numactl.
                    Omit to run unpinned on the whole machine.
  -j, --concurrency <n>
                    concurrent games. Default: CPUs on the node when --node is
                    given, else all CPUs available to this process. More than
                    that is an error.
  -b, --book <path> opening book (default: $BOOK)
  -f, --format <fmt>
                    book format: epd or pgn. Default: inferred from the book's
                    extension.
  -p, --plies <n>   plies to take from each PGN opening (default: $PLIES)
  -a, --adjudicate  end games early by resignation, draw and tablebase
                    adjudication. Default: off, to keep more endgame positions.
  -l, --log-dir <dir>
                    keep Sturddle engine logs in <dir>, created if missing.
                    Default: no engine logs.
  -n, --dry-run     print the command line and exit; no run, no files written
  -h, --help        this text

Env overrides:
  FASTCHESS ENGINE TB BOOK BOOK_FORMAT PLIES HASH DEPTH CONCURRENCY OUTDIR LOG_DIR SEED

Examples:
  ${0##*/}                       # unpinned, 1M games, all CPUs
  ${0##*/} --node 0              # pinned to node 0
  ${0##*/} --node 1 250000
  ${0##*/} -b books/8moves.pgn -p 16
  ${0##*/} -j 8 --dry-run
  ${0##*/} --adjudicate
USAGE
}

DRY_RUN=0
ADJUDICATE=0
NODE=
POSITIONAL=()
while (( $# )); do
    case $1 in
        -h|--help)         usage; exit 0 ;;
        -n|--dry-run)      DRY_RUN=1 ;;
        -a|--adjudicate)   ADJUDICATE=1 ;;
        -N|--node)         [[ ${2:-} ]] || die "--node needs a value"; NODE=$2; shift ;;
        --node=*)          NODE=${1#*=} ;;
        -j|--concurrency)  [[ ${2:-} ]] || die "--concurrency needs a value"; CONCURRENCY=$2; shift ;;
        --concurrency=*)   CONCURRENCY=${1#*=} ;;
        -b|--book)         [[ ${2:-} ]] || die "--book needs a value"; BOOK=$2; shift ;;
        --book=*)          BOOK=${1#*=} ;;
        -f|--format)       [[ ${2:-} ]] || die "--format needs a value"; BOOK_FORMAT=$2; shift ;;
        --format=*)        BOOK_FORMAT=${1#*=} ;;
        -p|--plies)        [[ ${2:-} ]] || die "--plies needs a value"; PLIES=$2; shift ;;
        --plies=*)         PLIES=${1#*=} ;;
        -l|--log-dir)      [[ ${2:-} ]] || die "--log-dir needs a value"; LOG_DIR=$2; shift ;;
        --log-dir=*)       LOG_DIR=${1#*=} ;;
        --)                shift; POSITIONAL+=("$@"); break ;;
        -*)                die "unknown option '$1' (try --help)" ;;
        *)                 POSITIONAL+=("$1") ;;
    esac
    shift
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

(( $# <= 1 )) || die "too many arguments (try --help)"
GAMES=${1:-$GAMES_DEFAULT}
[[ $GAMES =~ ^[0-9]+$ ]] || die "games must be an integer, got '$GAMES'"

[[ $PLIES =~ ^[0-9]+$ ]] || die "plies must be an integer, got '$PLIES'"
(( PLIES >= 1 )) || die "plies must be >= 1"

USE_NUMA=0
[[ -n $NODE ]] && USE_NUMA=1

[[ -r $BOOK ]] || die "opening book not readable at $BOOK"

if [[ -z ${BOOK_FORMAT:-} ]]; then
    case ${BOOK,,} in
        *.epd) BOOK_FORMAT=epd ;;
        *.pgn) BOOK_FORMAT=pgn ;;
        *) die "cannot infer book format from '${BOOK##*/}'; pass --format epd|pgn" ;;
    esac
fi
case $BOOK_FORMAT in
    epd|pgn) ;;
    *) die "book format must be epd or pgn, got '$BOOK_FORMAT'" ;;
esac

[[ -x $FASTCHESS ]] || die "fastchess not executable at $FASTCHESS"
[[ -x $ENGINE    ]] || die "engine not executable at $ENGINE"
[[ -d $TB        ]] || die "tablebase dir not found at $TB"
compgen -G "$TB/*.rtbw" >/dev/null || die "no .rtbw files in $TB"

if (( USE_NUMA )); then
    [[ $NODE =~ ^[0-9]+$ ]] || die "NUMA node must be an integer, got '$NODE'"
    command -v numactl >/dev/null || die "numactl not installed (omit --node to run unpinned)"

    AVAIL_NODES=$(numactl --hardware | awk '/^available:/ {print $2}')
    (( NODE < AVAIL_NODES )) || die "node $NODE out of range (system has $AVAIL_NODES node(s))"

    mapfile -t NODE_CPUS < <(numactl --hardware | awk -v n="$NODE" '$1=="node" && $2==n && $3=="cpus:" {for(i=4;i<=NF;i++) print $i}')
    MAX_CPU=${#NODE_CPUS[@]}
    (( MAX_CPU > 0 )) || die "could not determine CPU list for node $NODE"

    # --membind can't spill to other nodes, overcommitting gets OOM-killed
    FREE_MB=$(numactl --hardware | awk -v n="node $NODE free:" 'index($0,n)==1 {print $4}')
    WHERE="node $NODE"
else
    MAX_CPU=$(nproc)                            # respects taskset/cgroup affinity
    FREE_MB=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    WHERE="unpinned"
fi

CONCURRENCY=${CONCURRENCY:-$MAX_CPU}
[[ $CONCURRENCY =~ ^[0-9]+$ ]] || die "concurrency must be an integer, got '$CONCURRENCY'"
(( CONCURRENCY >= 1 )) || die "concurrency must be >= 1"
(( CONCURRENCY <= MAX_CPU )) || die "concurrency $CONCURRENCY exceeds $MAX_CPU available CPU(s) ($WHERE)"

WANT_MB=$(( HASH * 2 * CONCURRENCY ))           # two engines per game
if [[ -n ${FREE_MB:-} ]] && (( WANT_MB > FREE_MB * 85 / 100 )); then
    die "want ${WANT_MB}MB of hash ($HASH x 2 x $CONCURRENCY) but $WHERE has only ${FREE_MB}MB free"
fi

GUID=$(cat /proc/sys/kernel/random/uuid)
SEED=${SEED:-$(( (RANDOM << 15 | RANDOM) & 0x7fffffff ))}
PGN="$OUTDIR/train-data-$GUID.pgn"
LOG="$OUTDIR/train-data-$GUID.log"

ENGINE_NAME=${ENGINE##*/}
ENGINE_OPTS=()
ENGINE_LOG="-l /dev/null"
[[ -n $LOG_DIR ]] && ENGINE_LOG="-s -l '$LOG_DIR/sturddle.log'"
[[ $ENGINE_NAME == *sturddle* ]] && ENGINE_OPTS=(args="$ENGINE_LOG" option.OwnBook=false)

ADJUDICATE_OPTS=()
(( ADJUDICATE )) && ADJUDICATE_OPTS=(-resign movecount=10 score=700 twosided=true -draw movenumber=40 movecount=8 score=10 -tb "$TB" -tbadjudicate BOTH)

CMD=()
(( USE_NUMA )) && CMD+=(numactl --cpunodebind="$NODE" --membind="$NODE" --)
CMD+=("$FASTCHESS"
    -engine cmd="$ENGINE" name="$ENGINE_NAME-w" "${ENGINE_OPTS[@]+"${ENGINE_OPTS[@]}"}"
    -engine cmd="$ENGINE" name="$ENGINE_NAME-b" "${ENGINE_OPTS[@]+"${ENGINE_OPTS[@]}"}"
    -each proto=uci tc=inf depth="$DEPTH"
          option.Threads=1 option.Hash="$HASH"
          option.SyzygyPath="$TB"
    -rounds "$GAMES" -games 1 -repeat 1 -concurrency "$CONCURRENCY"
    -openings file="$BOOK" format="$BOOK_FORMAT" order=random plies="$PLIES"
    -srand "$SEED"
    "${ADJUDICATE_OPTS[@]+"${ADJUDICATE_OPTS[@]}"}"
    -recover
    -pgnout file="$PGN"
    -log file="$LOG" level=warn
)

echo "$WHERE ($MAX_CPU cpus) | concurrency $CONCURRENCY | $GAMES games | seed $SEED"
echo "book ${BOOK##*/} ($BOOK_FORMAT, $PLIES plies)"
echo "-> $PGN"

if (( DRY_RUN )); then
    printf 'dry run, nothing written:\n'
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

mkdir -p "$OUTDIR" ${LOG_DIR:+"$LOG_DIR"}
exec "${CMD[@]}"
