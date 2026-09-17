#!/usr/bin/env bash
# Persist the dedicated UR7e NetworkManager profile without activating it.
set -euo pipefail

readonly DEFAULT_CONNECTION="Wired connection 2"
readonly DEFAULT_HOST_CIDR="192.168.10.100/24"
readonly DEFAULT_ROBOT_IP="192.168.10.11"

MODE="check"
ASSUME_YES=0
CONNECTION="$DEFAULT_CONNECTION"
HOST_CIDR="$DEFAULT_HOST_CIDR"
ROBOT_IP="$DEFAULT_ROBOT_IP"
DEVICE=""

usage() {
  cat <<'EOF'
Usage: setup_ur7e_ethernet.sh [options]

Check or persist the isolated UR7e Ethernet profile. The default is read-only.

Options:
  --check                 Read-only verification (default).
  --apply --yes           Persist the requested NetworkManager profile settings.
                         This never activates/deactivates a connection.
  --device IFACE          Restrict all work to this Ethernet interface.
  --connection NAME       NetworkManager profile (default: Wired connection 2).
  --host-cidr CIDR        Host address (default: 192.168.10.100/24).
  --robot-ip ADDRESS      UR controller address (default: 192.168.10.11).
  -h, --help              Show this help.

Auto-detection accepts only an Ethernet NIC that already reaches the robot,
has the requested host address, is active on the selected profile, or matches
that profile's pinned MAC address. Ambiguous candidates fail closed.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

is_ipv4() {
  local address="$1" octet
  [[ "$address" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || return 1
  IFS=. read -r -a octets <<<"$address"
  for octet in "${octets[@]}"; do
    ((10#$octet <= 255)) || return 1
  done
}

is_ipv4_cidr() {
  local cidr="$1" address prefix
  address="${cidr%/*}"
  prefix="${cidr#*/}"
  is_ipv4 "$address" && [[ "$prefix" =~ ^[0-9]+$ ]] && ((10#$prefix <= 32))
}

contains_line() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    [[ "$value" == "$needle" ]] && return 0
  done
  return 1
}

is_ethernet() {
  [[ "$(nmcli -g GENERAL.TYPE device show "$1" 2>/dev/null || true)" == "ethernet" ]]
}

device_has_address() {
  local device="$1"
  local cidr="$2"
  ip -o -4 addr show dev "$device" 2>/dev/null | awk '{print $4}' | grep -Fxq "$cidr"
}

device_mac() {
  nmcli -g GENERAL.HWADDR device show "$1" | tr -d '\\' | tr '[:lower:]' '[:upper:]'
}

add_candidate() {
  local candidate="$1"
  [[ -n "$candidate" ]] || return
  is_ethernet "$candidate" || return
  contains_line "$candidate" "${CANDIDATES[@]}" || CANDIDATES+=("$candidate")
}

route_device() {
  ip route get "$ROBOT_IP" 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i == "dev") {print $(i + 1); exit}}'
}

discover_device() {
  local candidate active_connection profile_mac
  CANDIDATES=()

  if [[ -n "$DEVICE" ]]; then
    add_candidate "$DEVICE"
    [[ ${#CANDIDATES[@]} -eq 1 ]] || die "--device must name an existing Ethernet interface: $DEVICE"
    return
  fi

  candidate="$(route_device)"
  if [[ -n "$candidate" ]] && device_has_address "$candidate" "$HOST_CIDR"; then
    add_candidate "$candidate"
  fi

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    if timeout 1 ping -n -c 1 -W 1 -I "$candidate" "$ROBOT_IP" >/dev/null 2>&1; then
      add_candidate "$candidate"
    fi
  done < <(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2 == "ethernet" {print $1}')

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    active_connection="$(nmcli -g GENERAL.CONNECTION device show "$candidate" 2>/dev/null || true)"
    [[ "$active_connection" == "$CONNECTION" ]] && add_candidate "$candidate"
  done < <(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2 == "ethernet" {print $1}')

  profile_mac="$(nmcli -g 802-3-ethernet.mac-address connection show "$CONNECTION" 2>/dev/null | tr -d '\\' | tr '[:lower:]' '[:upper:]' || true)"
  if [[ -n "$profile_mac" && "$profile_mac" != "--" ]]; then
    while IFS= read -r candidate; do
      [[ -n "$candidate" ]] || continue
      [[ "$(device_mac "$candidate")" == "$profile_mac" ]] && add_candidate "$candidate"
    done < <(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2 == "ethernet" {print $1}')
  fi

  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    die "could not identify a dedicated UR7e Ethernet NIC; use --device IFACE"
  fi
  if [[ ${#CANDIDATES[@]} -ne 1 ]]; then
    printf 'ERROR: ambiguous Ethernet candidates:' >&2
    printf ' %s' "${CANDIDATES[@]}" >&2
    printf '\nUse --device IFACE to select the UR7e NIC.\n' >&2
    exit 1
  fi
  DEVICE="${CANDIDATES[0]}"
}

show_result() {
  local label="$1" actual="$2" expected="$3"
  if [[ "$actual" == "$expected" ]]; then
    printf 'OK    %-24s %s\n' "$label" "$actual"
  else
    printf 'WARN  %-24s actual=%q expected=%q\n' "$label" "$actual" "$expected"
    CHECK_FAILED=1
  fi
}

check_port() {
  local port="$1"
  if timeout 2 bash -c "</dev/tcp/$ROBOT_IP/$port" 2>/dev/null; then
    printf 'OK    robot TCP %-14s open\n' "$port"
  else
    printf 'WARN  robot TCP %-14s unavailable\n' "$port"
    CHECK_FAILED=1
  fi
}

check_configuration() {
  local actual addresses gateway active_connection expected_mac
  CHECK_FAILED=0
  expected_mac="$(device_mac "$DEVICE")"

  printf 'Profile: %s\nDevice:  %s (%s)\n' "$CONNECTION" "$DEVICE" "$expected_mac"
  actual="$(nmcli -g connection.interface-name connection show "$CONNECTION")"
  show_result "profile interface" "$actual" "$DEVICE"
  actual="$(nmcli -g 802-3-ethernet.mac-address connection show "$CONNECTION" | tr -d '\\' | tr '[:lower:]' '[:upper:]')"
  show_result "profile MAC" "$actual" "$expected_mac"
  show_result "IPv4 method" "$(nmcli -g ipv4.method connection show "$CONNECTION")" "manual"
  addresses="$(nmcli -g ipv4.addresses connection show "$CONNECTION")"
  if printf '%s\n' "$addresses" | tr ',' '\n' | grep -Fxq "$HOST_CIDR"; then
    printf 'OK    %-24s %s\n' "IPv4 address" "$HOST_CIDR"
  else
    printf 'WARN  %-24s missing %s (actual=%q)\n' "IPv4 address" "$HOST_CIDR" "$addresses"
    CHECK_FAILED=1
  fi
  gateway="$(nmcli -g ipv4.gateway connection show "$CONNECTION")"
  show_result "IPv4 gateway" "$gateway" ""
  show_result "IPv4 static routes" "$(nmcli -g ipv4.routes connection show "$CONNECTION")" ""
  show_result "IPv4 never-default" "$(nmcli -g ipv4.never-default connection show "$CONNECTION")" "yes"
  show_result "IPv6 method" "$(nmcli -g ipv6.method connection show "$CONNECTION")" "disabled"

  active_connection="$(nmcli -g GENERAL.CONNECTION device show "$DEVICE" 2>/dev/null || true)"
  show_result "active profile" "$active_connection" "$CONNECTION"
  if device_has_address "$DEVICE" "$HOST_CIDR"; then
    printf 'OK    %-24s %s\n' "live IPv4 address" "$HOST_CIDR"
  else
    printf 'WARN  %-24s missing %s\n' "live IPv4 address" "$HOST_CIDR"
    CHECK_FAILED=1
  fi
  if ip route get "$ROBOT_IP" 2>/dev/null | grep -Fq "dev $DEVICE"; then
    printf 'OK    %-24s %s via %s\n' "robot route" "$ROBOT_IP" "$DEVICE"
  else
    printf 'WARN  %-24s %s is not routed via %s\n' "robot route" "$ROBOT_IP" "$DEVICE"
    CHECK_FAILED=1
  fi
  check_port 29999
  check_port 30001
  check_port 30004
  return "$CHECK_FAILED"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) MODE="check" ;;
    --apply) MODE="apply" ;;
    --yes) ASSUME_YES=1 ;;
    --device|--connection|--host-cidr|--robot-ip)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --device) DEVICE="$2" ;;
        --connection) CONNECTION="$2" ;;
        --host-cidr) HOST_CIDR="$2" ;;
        --robot-ip) ROBOT_IP="$2" ;;
      esac
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

need_command nmcli
need_command ip
need_command ping
need_command timeout
[[ "$HOST_CIDR" == */* ]] || die "--host-cidr must include a prefix length"
is_ipv4_cidr "$HOST_CIDR" || die "--host-cidr must be an IPv4 CIDR"
is_ipv4 "$ROBOT_IP" || die "--robot-ip must be an IPv4 address"
nmcli connection show "$CONNECTION" >/dev/null 2>&1 || die "NetworkManager profile not found: $CONNECTION"
discover_device

if [[ "$MODE" == "apply" ]]; then
  [[ $ASSUME_YES -eq 1 ]] || die "--apply requires --yes"
  if [[ ${EUID} -ne 0 ]]; then
    die "--apply needs NetworkManager privileges; rerun this script with sudo"
  fi
  nmcli connection modify "$CONNECTION" \
    connection.interface-name "$DEVICE" \
    802-3-ethernet.mac-address "$(device_mac "$DEVICE")" \
    ipv4.method manual \
    ipv4.addresses "$HOST_CIDR" \
    ipv4.gateway "" \
    ipv4.routes "" \
    ipv4.never-default yes \
    ipv6.method disabled
  printf 'Persisted %s for %s. The active connection was not restarted.\n' "$CONNECTION" "$DEVICE"
fi

if check_configuration; then
  exit 0
fi
exit 2
