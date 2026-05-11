#!/bin/bash
# pg-tpch (yuhong-zhong/pg-tpch): the tpch_prepare script assumes a custom
# PostgreSQL 9.3 built from source at /usr/local/{bin,lib}. pg-tpch needs 9.3
# specifically because tpch_prepare writes `checkpoint_segments` to
# postgresql.conf, a GUC that was removed in 9.5.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

PG_VERSION=9.3.25

# Build deps. `bison`/`flex` for the parser; `libreadline-dev` for psql line
# editing; `zlib1g-dev` for pg_dump compression; `libxslt1-dev`/`libxml2-dev`
# for XML support. `gcc-10` on focal is fine but it defaults to `-fno-common`,
# which breaks postgres 9.3's multiple-definition style -- handled by CFLAGS.
sudo apt-get install -y \
    build-essential bison flex \
    libreadline-dev zlib1g-dev libxslt1-dev libxml2-dev libssl-dev

# Increase SHM limits so postgres can allocate shared buffers.
sudo sysctl -w kernel.shmmax=2147483648 >/dev/null
sudo sysctl -w kernel.shmall=2097152 >/dev/null

cd "$BASE_PATH"

# Build + install PostgreSQL 9.3 to /usr/local if pg_ctl isn't there yet.
# Local build artifacts go into BASE_PATH (the shared mount) so other VMs see
# the source tree, but `make install` writes to /usr/local on the VM's local
# filesystem -- each VM still needs to run `make install` once.
if [ ! -x /usr/local/bin/pg_ctl ]; then
    if [ ! -d "postgresql-${PG_VERSION}" ]; then
        wget -q "https://ftp.postgresql.org/pub/source/v${PG_VERSION}/postgresql-${PG_VERSION}.tar.gz"
        tar -xzf "postgresql-${PG_VERSION}.tar.gz"
    fi
    cd "postgresql-${PG_VERSION}"
    # gcc-10+ defaults to -fno-common, which breaks postgres 9.3's reliance on
    # tentative-definition merging across compilation units. -fcommon restores
    # the pre-10 behavior. Also suppress -Werror on warnings that became
    # errors-by-default in newer gcc.
    #
    # The build outputs (object files, postgres binary) live in the shared
    # source tree, so once vm1 has done the configure+make, every other VM can
    # just run `sudo make install` against the same tree to copy binaries onto
    # its own local /usr/local. Concurrent configure runs would race on the
    # shared tree -- guard with a prebuilt-binary check.
    if [ ! -x src/backend/postgres ]; then
        CFLAGS="-O2 -fno-omit-frame-pointer -fcommon -Wno-error=incompatible-pointer-types -Wno-error=int-conversion" \
            ./configure --prefix=/usr/local --enable-debug >/dev/null
        make -j"$(nproc)"
    fi
    sudo make install
    cd "$BASE_PATH"
fi

if [ ! -d "pg-tpch" ]; then
    git clone https://github.com/yuhong-zhong/pg-tpch.git
fi
cd pg-tpch

# Build the TPC-H toolkit (dbgen + qgen) in the shared dir. These are the only
# binary deps each VM needs at install time; initdb / COPY / index / analyze
# (the workload-data prep that tpch_prepare normally does) is deferred to
# benchmark run-time.
if [ ! -x dbgen/dbgen ] || [ ! -x dbgen/qgen ]; then
    (cd dbgen && make -j"$(nproc)" dbgen qgen)
fi

# Patch pg-tpch's pgtpch_defaults so a later `./tpch_prepare` (run from the
# benchmark driver) reuses the pre-generated .tbl files on the shared mount
# instead of regenerating per-VM:
#   - TPCHTMP -> shared path  (avoids 11 GB local disk + 3 min dbgen per VM)
#   - REMAKE_DATA -> false    (dbgen output already exists in shared)
#   - skip rm -rf "$TPCHTMP"  (next VM still needs the .tbl files)
SHARED_TPCHTMP=/home/ubuntu/shared/tpch_tmp
sed -i "s|^TPCHTMP=.*|TPCHTMP=$SHARED_TPCHTMP|" pgtpch_defaults
sed -i "s|^REMAKE_DATA=.*|REMAKE_DATA=false|"   pgtpch_defaults
sed -i 's|sudo -u \$PGUSER rm -rf "\$TPCHTMP"|true # patched: keep .tbl files for other VMs|' tpch_prepare

# Pre-generate .tbl files in the shared dir under flock so the first VM to
# arrive runs dbgen (~3 min) and the rest skip. Without this, the later
# tpch_prepare run would either race or fall back to per-VM dbgen.
mkdir -p "$SHARED_TPCHTMP"
(
    flock -x 9
    if [ ! -f "$SHARED_TPCHTMP/lineitem.tbl" ]; then
        cp "$BASE_PATH/pg-tpch/dbgen/dists.dss" "$SHARED_TPCHTMP/"
        cd "$SHARED_TPCHTMP"
        for t in c s n r O L P S; do
            "$BASE_PATH/pg-tpch/dbgen/dbgen" -s 10 -f -v -T $t &
        done
        wait
    fi
) 9>"/home/ubuntu/shared/tpch_tmp.lock"
