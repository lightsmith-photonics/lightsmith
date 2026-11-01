#!/usr/bin/env bash
# Build the documentation and publish it to docs.lightsmith.dev (nginx on the Lightsmith server).
#
#   scripts/deploy_docs.sh                 # needs SSH access as root@95.217.87.43
#   DOCS_HOST=user@host scripts/deploy_docs.sh
#
# Each deploy is unpacked into its own release directory and switched in atomically by
# repointing the `current` symlink, which nginx serves; the last five releases are kept.
set -euo pipefail

host="${DOCS_HOST:-root@95.217.87.43}"
base=/var/www/lightsmith-docs
release="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"

cd "$(dirname "$0")/.."
NO_MKDOCS_2_WARNING=1 uv run --group docs mkdocs build --strict
tar -C site -czf - . | ssh "$host" "set -e
  mkdir -p $base/releases/$release
  tar -xzf - -C $base/releases/$release
  ln -sfn releases/$release $base/current.tmp && mv -T $base/current.tmp $base/current
  cd $base/releases && ls -1 | sort | head -n -5 | xargs -r rm -rf"
echo "deployed $release to https://docs.lightsmith.dev/"
