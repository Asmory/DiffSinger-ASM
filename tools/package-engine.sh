#!/usr/bin/env bash
set -euo pipefail

version=${1:?usage: package-engine.sh VERSION [OUTPUT_DIR]}
output_dir=${2:-release}

if [[ ! $version =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid package version: $version" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case $output_dir in
  /*) ;;
  *) output_dir="$repo_root/$output_dir" ;;
esac

package_name="diffsinger-asm-${version}-linux-x86_64"
archive="$output_dir/$package_name.tar.gz"
checksum="$archive.sha256"
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

install -d \
  "$staging/$package_name/bin" \
  "$staging/$package_name/lib" \
  "$staging/$package_name/include" \
  "$staging/$package_name/config" \
  "$staging/$package_name/docs"
install -m 0755 "$repo_root/build/dsasm-acoustic" "$staging/$package_name/bin/"
install -m 0755 "$repo_root/build/dsasm-vocoder-m40" "$staging/$package_name/bin/"
install -m 0755 "$repo_root/build/libdsasm.so" "$staging/$package_name/lib/"
install -m 0644 "$repo_root/include/dsasm_engine.h" "$staging/$package_name/include/"
install -m 0644 "$repo_root/config/best-inference.env" "$staging/$package_name/config/"
install -m 0644 "$repo_root/README.md" "$staging/$package_name/"
install -m 0644 "$repo_root/docs/ENGINE_ABI.md" "$staging/$package_name/docs/"
install -m 0644 "$repo_root/docs/PERFORMANCE.md" "$staging/$package_name/docs/"

mkdir -p "$output_dir"
source_date_epoch=${SOURCE_DATE_EPOCH:-$(git -C "$repo_root" log -1 --format=%ct)}
tar --sort=name \
  --mtime="@$source_date_epoch" \
  --owner=0 --group=0 --numeric-owner \
  -C "$staging" -cf - "$package_name" | gzip -n > "$archive"
(
  cd "$output_dir"
  sha256sum "$(basename "$archive")" > "$(basename "$checksum")"
)
printf 'Created %s\nCreated %s\n' "$archive" "$checksum"
