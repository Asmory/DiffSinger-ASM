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
python_runtime=${DSASM_PYTHON_RUNTIME:-}
if [[ -z $python_runtime ]]; then
  command -v uv >/dev/null || { echo 'uv is required to assemble the self-contained model tool' >&2; exit 2; }
  python_executable=$(uv python find 3.12)
  python_runtime=$(readlink -f "$(dirname "$python_executable")/..")
fi
test -x "$python_runtime/bin/python3.12" || { echo "invalid CPython 3.12 runtime: $python_runtime" >&2; exit 2; }

package_cache=${DSASM_PACKAGE_CACHE_DIR:-$repo_root/build/package-cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$package_cache/uv}
requirements_sha=$(sha256sum "$repo_root/tools/model-tool-requirements.txt" | cut -d' ' -f1)
dependency_cache="$package_cache/model-tool-$requirements_sha"
if [[ ! -f $dependency_cache/.complete ]]; then
  dependency_build="$package_cache/.model-tool-$requirements_sha.building.$$"
  install -d "$dependency_build/site-packages"
  trap 'rm -rf "$staging" "${dependency_build:-}"' EXIT
  uv pip install --target "$dependency_build/site-packages" \
    --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
    --require-hashes --no-deps -r "$repo_root/tools/model-tool-requirements.txt"
  : > "$dependency_build/.complete"
  if ! mv "$dependency_build" "$dependency_cache" 2>/dev/null; then
    test -f "$dependency_cache/.complete"
    rm -rf "$dependency_build"
  fi
fi

install -d \
  "$staging/$package_name/bin" \
  "$staging/$package_name/lib" \
  "$staging/$package_name/include" \
  "$staging/$package_name/config" \
  "$staging/$package_name/docs" \
  "$staging/$package_name/share/dsasm-model-tool/app"
install -m 0755 "$repo_root/build/dsasm-acoustic" "$staging/$package_name/bin/"
install -m 0755 "$repo_root/build/dsasm-vocoder-m40" "$staging/$package_name/bin/"
install -m 0755 "$repo_root/build/libdsasm.so" "$staging/$package_name/lib/"
install -m 0644 "$repo_root/include/dsasm_engine.h" "$staging/$package_name/include/"
install -m 0644 "$repo_root/config/best-inference.env" "$staging/$package_name/config/"
install -m 0644 "$repo_root/README.md" "$staging/$package_name/"
install -m 0644 "$repo_root/docs/ENGINE_ABI.md" "$staging/$package_name/docs/"
install -m 0644 "$repo_root/docs/PERFORMANCE.md" "$staging/$package_name/docs/"
install -m 0644 "$repo_root/docs/OPENUTAU_OFFLINE_MODEL_PROTOCOL.md" "$staging/$package_name/docs/"
install -m 0755 "$repo_root/tools/dsasm-model-tool" "$staging/$package_name/bin/"
install -m 0644 "$repo_root/tools/dsasm_model_tool.py" "$staging/$package_name/share/dsasm-model-tool/app/"
install -m 0644 "$repo_root/tools/pack_acoustic_onnx_m25.py" "$staging/$package_name/share/dsasm-model-tool/app/"
install -m 0644 "$repo_root/tools/pack_vocoder_graph_m35.py" "$staging/$package_name/share/dsasm-model-tool/app/"
install -m 0644 "$repo_root/tools/dsv35_common.py" "$staging/$package_name/share/dsasm-model-tool/app/"
install -m 0644 "$repo_root/tools/model-tool-requirements.txt" "$staging/$package_name/share/dsasm-model-tool/"
provider_commit=$(git -C "$repo_root" rev-parse HEAD)
provider_dirty=
if [[ -n $(git -C "$repo_root" status --porcelain) ]]; then
  provider_dirty=-dirty
fi
printf 'package:%s+git:%s%s\n' "$version" "$provider_commit" "$provider_dirty" > \
  "$staging/$package_name/share/dsasm-model-tool/provider-build-id"
cp -a "$python_runtime" "$staging/$package_name/share/dsasm-model-tool/python"
install -d "$staging/$package_name/share/dsasm-model-tool/python/lib/python3.12/site-packages"
cp -a "$dependency_cache/site-packages/." \
  "$staging/$package_name/share/dsasm-model-tool/python/lib/python3.12/site-packages/"
"$staging/$package_name/bin/dsasm-model-tool" --help >/dev/null

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
