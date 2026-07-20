#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${1:-${repo_root}/artifacts/vendor_assets/mid360}"
cad_url="https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Mid360/mid-360-asm.stp"
cad_sha256="b93e9b51282ed319b6aa755e76a132c0eb03306da5f3b9676bcabf2e2ae25f02"
target_file="${target_dir}/mid-360-asm.stp"

mkdir -p "${target_dir}"
temporary_file="$(mktemp "${target_dir}/.mid360-cad.XXXXXX")"
trap 'rm -f "${temporary_file}"' EXIT

curl --fail --location --retry 5 --retry-all-errors \
  --connect-timeout 20 --output "${temporary_file}" "${cad_url}"

actual_sha256="$(sha256sum "${temporary_file}" | cut -d ' ' -f 1)"
if [[ "${actual_sha256}" != "${cad_sha256}" ]]; then
  echo "Mid-360 CAD checksum mismatch: ${actual_sha256}" >&2
  exit 1
fi

mv "${temporary_file}" "${target_file}"
trap - EXIT
echo "Verified official Mid-360 CAD: ${target_file}"
