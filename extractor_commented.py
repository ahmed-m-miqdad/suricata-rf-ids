#!/usr/bin/env python3
"""
extractor.py — Service 1: Feature Extractor
============================================

WHAT THIS SERVICE DOES:
    This is the FIRST service in the hybrid IDS pipeline.
    It runs continuously in the background, watching Suricata's eve.json log file.

    Suricata writes a JSON record for EVERY network flow it sees — both normal
    traffic and attacks. This service reads those records and splits them into
    two paths:

    PATH 1 — Suricata already recognised it (alert event):
        → Log it directly to alerts_log.csv as KNOWN ATTACK
        → No machine learning needed — Suricata's signature matched

    PATH 2 — Suricata did NOT recognise it (unalerted flow event):
        → Extract 19 numerical features from the flow record
        → Write those features to flows.csv
        → detector.py will pick them up and run the RF model

HOW IT FITS IN THE PIPELINE:
    Suricata → eve.json → [extractor.py] → flows.csv → [detector.py] → traffic.log
                                        ↘ alerts_log.csv (known attacks)

Usage:
    python extractor.py [--eve /var/log/suricata/eve.json]
                        [--queue /opt/ids/flows.csv]
                        [--alerts /opt/ids/alerts_log.csv]

Version notes (important changes that were fixed during development):
    - Added tcp_flags_client (feature #14) — was missing in v1, model expects 19
    - Fixed flow_duration units: now in MICROSECONDS to match CICIDS2017 training
      (v1 used seconds → flow_bytes_per_sec was computed on wrong scale)
    - Updated FEATURE_NAMES from 18 → 19 features
"""

import os
import sys
import csv
import json
import time
import fcntl        # Linux file locking — prevents race conditions between services
import logging
import argparse
import syslog       # Linux system log — for centralised security event logging
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration constants ───────────────────────────────────────────────────
# These are the default paths. Can be overridden with command-line arguments.
DEFAULT_EVE    = '/var/log/suricata/eve.json'  # Suricata's output log
DEFAULT_QUEUE  = '/opt/ids/flows.csv'           # Queue file for detector.py
DEFAULT_ALERTS = '/opt/ids/alerts_log.csv'      # Known-attack log
POLL_INTERVAL  = 0.5   # How often (in seconds) to check eve.json for new lines

# ── 19-feature contract ───────────────────────────────────────────────────────
# This list defines which features to extract and in what ORDER.
# The order here must be identical to what was used during model training
# (saved in feature_contract.pkl). If the order changes, the model gets
# the wrong values for the wrong features and predictions will be garbage.
FEATURE_NAMES = [
    'destination_port',      # Port the traffic is aimed at (22=SSH, 80=HTTP, 443=HTTPS…)
    'flow_duration',         # How long the connection lasted — stored in MICROSECONDS
                             # IMPORTANT: CICIDS2017 dataset uses microseconds,
                             # so we must convert Suricata's timestamps to microseconds
                             # to keep the scale consistent with training data.
    'pkts_toserver',         # Number of packets sent FROM client TO server
    'pkts_toclient',         # Number of packets sent FROM server TO client
    'bytes_toserver',        # Total bytes sent from client to server
    'bytes_toclient',        # Total bytes sent from server to client
    'flow_bytes_per_sec',    # DERIVED: (total_bytes) / (duration in seconds)
                             # Falls back to 0.0 if duration is 0 (instant flows)
    'flow_pkts_per_sec',     # DERIVED: (total_packets) / (duration in seconds)
                             # Falls back to 0.0 if duration is 0
    'down_up_ratio',         # DERIVED: pkts_toclient / pkts_toserver
                             # How much did the server respond relative to client?
                             # High value = server sent a lot (e.g. file download)
                             # Falls back to 0.0 if pkts_toserver is 0
    'fin_flag',              # Was the TCP FIN flag set? 1=yes, 0=no
                             # FIN = connection was closed normally
    'syn_flag',              # Was the TCP SYN flag set? 1=yes, 0=no
                             # SYN = connection was being opened (lots of SYNs = port scan)
    'psh_flag',              # Was the TCP PSH flag set? 1=yes, 0=no
                             # PSH = "send this data now" (common in HTTP/interactive traffic)
    'ack_flag',              # Was the TCP ACK flag set? 1=yes, 0=no
                             # ACK = acknowledging received data (present in most flows)
    'tcp_flags_client',      # Client's combined TCP flags as a single integer
                             # Suricata stores this as a hex string in tcp_flags_ts
                             # e.g. '1a' hex → 26 decimal → bits 1,3,4 set (SYN+PSH+ACK)
                             # Was MISSING in v1 — caused 18-feature mismatch with model
    'total_bytes',           # DERIVED: bytes_toserver + bytes_toclient
    'total_packets',         # DERIVED: pkts_toserver + pkts_toclient
    'fwd_bytes_per_pkt',     # DERIVED: bytes_toserver / pkts_toserver
                             # Average size of each packet going TO the server
                             # Falls back to 0.0 if pkts_toserver is 0
    'bwd_bytes_per_pkt',     # DERIVED: bytes_toclient / pkts_toclient
                             # Average size of each packet coming FROM the server
                             # Falls back to 0.0 if pkts_toclient is 0
    'bytes_ratio',           # DERIVED: bytes_toserver / total_bytes
                             # What fraction of bytes went TO the server?
                             # Range: 0.0 (all download) to 1.0 (all upload)
                             # Falls back to 0.5 (NOT 0.0) when total_bytes=0
                             # because 0.5 = perfectly symmetric = correct for empty flow
]

# Queue CSV has metadata columns first, then the 19 feature columns.
# Metadata is used for logging and alerting — not fed to the model.
QUEUE_COLUMNS = ['flow_id', 'timestamp', 'src_ip', 'src_port',
                 'dest_ip', 'dest_port', 'proto'] + FEATURE_NAMES

# ── ANSI terminal colour codes ────────────────────────────────────────────────
RED    = '\033[91m'   # Used for KNOWN ATTACK messages
YELLOW = '\033[93m'
GREEN  = '\033[92m'
CYAN   = '\033[96m'
RESET  = '\033[0m'   # Resets colour back to terminal default

# ── Python logging setup ──────────────────────────────────────────────────────
# Logs timestamped messages to stdout (captured by systemd journal)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('extractor')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: timestamp parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_ts(ts_str):
    """
    Convert a Suricata ISO 8601 timestamp string into a Python datetime object.

    Suricata timestamps look like: '2026-03-22T09:00:00.123456+0000'
    We need Python datetime objects so we can subtract start from end to get
    the flow duration in seconds (then multiply by 1,000,000 for microseconds).

    The 'Z' suffix (meaning UTC) is not valid in Python's fromisoformat() before
    Python 3.11, so we replace it with '+00:00' for compatibility.

    Returns None on failure so the caller can use the 'age' fallback instead.
    """
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except Exception:
        return None   # Caller will use flow.get('age', 0) as fallback


# ─────────────────────────────────────────────────────────────────────────────
# Core function: extract 19 features from one Suricata flow event
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(event):
    """
    Given one parsed Suricata eve.json 'flow' event (as a Python dict),
    extract and return all 19 model features as a dict.

    A Suricata flow event looks roughly like this in eve.json:
    {
      "event_type": "flow",
      "src_ip": "192.168.240.129",
      "src_port": 54840,
      "dest_ip": "192.168.240.133",
      "dest_port": 22,
      "proto": "TCP",
      "flow": {
        "pkts_toserver": 15,
        "pkts_toclient": 12,
        "bytes_toserver": 2450,
        "bytes_toclient": 3100,
        "start": "2026-03-22T09:00:00.000000+0000",
        "end":   "2026-03-22T09:00:12.500000+0000",
        "age": 12,
        "alerted": false
      },
      "tcp": {
        "syn": true,
        "ack": true,
        "fin": true,
        "psh": false,
        "tcp_flags_ts": "1b"
      }
    }

    CRITICAL UNIT AND FALLBACK RULES (must exactly match preprocessing notebook):
      flow_duration      → MICROSECONDS  (seconds × 1,000,000)
      flow_bytes_per_sec → uses duration in SECONDS (not microseconds)
      flow_pkts_per_sec  → uses duration in SECONDS
      down_up_ratio      → 0.0 if pkts_toserver == 0
      fwd_bytes_per_pkt  → 0.0 if pkts_toserver == 0
      bwd_bytes_per_pkt  → 0.0 if pkts_toclient == 0
      bytes_ratio        → 0.5 if total_bytes == 0  (NOT 0.0 — symmetric assumption)
      tcp_flags_client   → 0   if tcp_flags_ts field missing or unparseable

    Returns:
        dict of {feature_name: value} matching FEATURE_NAMES order, or
        None if anything goes wrong (caller skips this flow)
    """
    try:
        # Pull out the nested sub-objects from the event
        flow = event.get('flow', {})
        tcp  = event.get('tcp',  {})

        # ── Step 1: Read direct fields from Suricata ──────────────────────────
        # These map 1-to-1 from Suricata eve.json fields to CICIDS2017 columns
        dest_port      = int(event.get('dest_port', 0))
        pkts_toserver  = int(flow.get('pkts_toserver', 0))
        pkts_toclient  = int(flow.get('pkts_toclient', 0))
        bytes_toserver = float(flow.get('bytes_toserver', 0))
        bytes_toclient = float(flow.get('bytes_toclient', 0))

        # ── Step 2: Compute flow_duration in MICROSECONDS ─────────────────────
        # WHY MICROSECONDS?
        # The CICIDS2017 dataset was built by CICFlowMeter, which stores
        # flow_duration in microseconds. Our model trained on that column.
        # If we stored duration in seconds here, the model would see values
        # 1,000,000× smaller than what it learned from → wrong predictions.
        #
        # Primary path: parse start/end ISO timestamps and compute difference
        t_start = parse_ts(flow.get('start', ''))
        t_end   = parse_ts(flow.get('end',   ''))
        if t_start and t_end:
            # total_seconds() returns a float, e.g. 12.5 for a 12.5-second flow
            duration_secs = max((t_end - t_start).total_seconds(), 0.0)
            flow_duration = duration_secs * 1_000_000   # convert to microseconds
        else:
            # Fallback: use Suricata's 'age' field (already in seconds)
            duration_secs = float(flow.get('age', 0))
            flow_duration = duration_secs * 1_000_000

        # ── Step 3: Extract TCP flag fields ───────────────────────────────────
        # Each flag is True/False in the 'tcp' object.
        # We encode as 1/0 integers to match CICIDS2017's flag count columns.
        fin_flag = 1 if tcp.get('fin') else 0   # Connection tear-down
        syn_flag = 1 if tcp.get('syn') else 0   # Connection initiation
        psh_flag = 1 if tcp.get('psh') else 0   # Push data immediately
        ack_flag = 1 if tcp.get('ack') else 0   # Acknowledgement

        # tcp_flags_client: Suricata stores the combined client-direction flags
        # as a hex string in 'tcp_flags_ts' (ts = "to server" direction).
        # We convert hex → integer so the model gets a numeric value.
        # Example: '1b' hex = 00011011 binary = FIN+SYN+PSH+ACK all set = 27
        try:
            tcp_flags_client = int(tcp.get('tcp_flags_ts', '00'), 16)
        except (ValueError, TypeError):
            tcp_flags_client = 0   # Safe default if field is absent or malformed

        # ── Step 4: Compute derived features ──────────────────────────────────
        # These are calculated from the direct fields above
        total_bytes   = bytes_toserver + bytes_toclient
        total_packets = pkts_toserver  + pkts_toclient

        # Rate features: how fast was data moving?
        # Note: we divide by duration_SECS (not flow_duration in µs)
        # because bytes-per-SECOND is the meaningful unit.
        # flow_duration (µs) is stored separately in the feature vector.
        flow_bytes_per_sec = (total_bytes   / duration_secs) if duration_secs > 0 else 0.0
        flow_pkts_per_sec  = (total_packets / duration_secs) if duration_secs > 0 else 0.0

        # Traffic asymmetry: did the server talk back as much as the client?
        down_up_ratio = (pkts_toclient / pkts_toserver) if pkts_toserver > 0 else 0.0

        # Average packet sizes in each direction
        fwd_bytes_per_pkt = (bytes_toserver / pkts_toserver) if pkts_toserver > 0 else 0.0
        bwd_bytes_per_pkt = (bytes_toclient / pkts_toclient) if pkts_toclient > 0 else 0.0

        # What fraction of traffic went TO the server?
        # 0.5 fallback (not 0.0) because a zero-byte flow is symmetric by definition
        bytes_ratio = (bytes_toserver / total_bytes) if total_bytes > 0 else 0.5

        # ── Step 5: Return all 19 features as a dict ──────────────────────────
        # The dict keys must match FEATURE_NAMES exactly.
        # Rounding keeps the CSV file readable and avoids floating point noise.
        return {
            'destination_port'  : dest_port,
            'flow_duration'     : round(flow_duration, 2),       # microseconds
            'pkts_toserver'     : pkts_toserver,
            'pkts_toclient'     : pkts_toclient,
            'bytes_toserver'    : bytes_toserver,
            'bytes_toclient'    : bytes_toclient,
            'flow_bytes_per_sec': round(flow_bytes_per_sec, 4),
            'flow_pkts_per_sec' : round(flow_pkts_per_sec,  4),
            'down_up_ratio'     : round(down_up_ratio,      4),
            'fin_flag'          : fin_flag,
            'syn_flag'          : syn_flag,
            'psh_flag'          : psh_flag,
            'ack_flag'          : ack_flag,
            'tcp_flags_client'  : tcp_flags_client,
            'total_bytes'       : total_bytes,
            'total_packets'     : total_packets,
            'fwd_bytes_per_pkt' : round(fwd_bytes_per_pkt, 4),
            'bwd_bytes_per_pkt' : round(bwd_bytes_per_pkt, 4),
            'bytes_ratio'       : round(bytes_ratio,        6),
        }

    except Exception as e:
        log.warning(f'Feature extraction failed: {e}')
        return None   # Signal to the caller to skip this flow


# ─────────────────────────────────────────────────────────────────────────────
# Queue writer: appends feature rows to flows.csv
# ─────────────────────────────────────────────────────────────────────────────

def queue_write(queue_path, rows):
    """
    Append a batch of flow feature rows to flows.csv.

    WHY BATCHING?
    We collect all new flows from one iteration of the main loop and write
    them together. This is more efficient than writing one row at a time
    because each write requires acquiring a file lock.

    WHY FILE LOCKING (fcntl)?
    Two processes touch flows.csv simultaneously:
      - extractor.py  WRITES new rows
      - detector.py   READS and CLEARS the file
    Without a lock, they could collide — extractor might write half a row
    exactly while detector is reading, producing a corrupted CSV line.
    fcntl.LOCK_EX = exclusive lock: blocks all other lock requests until released.

    Args:
        queue_path: path to flows.csv
        rows: list of dicts, each containing QUEUE_COLUMNS keys
    """
    if not rows:
        return   # Nothing to write — skip the lock entirely

    queue_path = Path(queue_path)
    # Only write the CSV header if the file is new or was just cleared
    write_header = not queue_path.exists() or queue_path.stat().st_size == 0

    lock_path = str(queue_path) + '.lock'   # Companion lock file

    try:
        with open(lock_path, 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)   # Acquire exclusive lock — blocks until free

            with open(queue_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=QUEUE_COLUMNS,
                                        extrasaction='ignore')
                if write_header:
                    writer.writeheader()   # Write column names as first row
                writer.writerows(rows)     # Write all flow rows

            fcntl.flock(lf, fcntl.LOCK_UN)   # Release lock so detector.py can read

    except Exception as e:
        log.error(f'Queue write failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Alert writer: logs Suricata signature matches to alerts_log.csv
# ─────────────────────────────────────────────────────────────────────────────

# Columns written for each known-attack alert entry
ALERT_COLUMNS = ['timestamp', 'type', 'src_ip', 'src_port',
                 'dest_ip', 'dest_port', 'proto',
                 'alert_signature', 'alert_category', 'alert_severity',
                 'confidence', 'flow_id']

def write_alert(alerts_path, row):
    """
    Append one KNOWN ATTACK entry to alerts_log.csv.
    Called whenever Suricata fires an 'alert' event (signature match).
    Confidence is set to 1.0 because a signature match is definitive.
    """
    alerts_path = Path(alerts_path)
    write_header = not alerts_path.exists() or alerts_path.stat().st_size == 0
    try:
        with open(alerts_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ALERT_COLUMNS,
                                    extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        log.error(f'Alert write failed: {e}')


def send_syslog(priority, message):
    """
    Write a message to the Linux system log (syslog).

    WHY SYSLOG?
    Syslog is the standard centralised logging facility on Linux.
    Messages written here appear in /var/log/syslog and can be:
      - Forwarded to a remote SIEM (Security Information & Event Management) system
      - Picked up by tools like Logstash, Splunk, or Graylog
      - Monitored in real time with: journalctl -f -t ids-extractor

    LOG_LOCAL0 = custom application facility (avoids mixing with OS logs)
    LOG_WARNING = used for actual attacks
    LOG_INFO    = used for normal/informational events
    """
    try:
        syslog.openlog('ids-extractor', syslog.LOG_PID, syslog.LOG_LOCAL0)
        syslog.syslog(priority, message)
        syslog.closelog()
    except Exception as e:
        log.warning(f'Syslog send failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Main loop: continuously tail eve.json and process new events
# ─────────────────────────────────────────────────────────────────────────────

def watch_eve(eve_path, queue_path, alerts_path):
    """
    The main event loop — runs forever until killed.

    HOW TAILING WORKS:
    Rather than re-reading the entire eve.json file every half-second,
    we track our current position (byte offset) in the file.
    Each iteration we seek() to that position and only read new lines
    written since our last check. This is exactly how 'tail -f' works.

    HOW LOG ROTATION IS HANDLED:
    Suricata periodically rotates eve.json (renames it and creates a new empty one).
    We detect this by checking the file's inode number — a unique identifier the OS
    assigns to each file. If the inode changes, the file was rotated and we
    reset our position to 0 to start reading the new file from the beginning.

    EVENT ROUTING:
    Each line in eve.json is a JSON object with an 'event_type' field:
      "alert" → Suricata signature matched → log as KNOWN ATTACK immediately
      "flow"  → Flow completed (no signature match) → extract features → queue
      others  → Ignored (dns, http, tls metadata events etc.)
    """
    eve_path = Path(eve_path)
    log.info(f'Watching : {eve_path}')
    log.info(f'Queue    : {queue_path}')
    log.info(f'Alerts   : {alerts_path}')
    log.info(f'Features : {len(FEATURE_NAMES)} (contract order)')
    log.info('Extractor running — Ctrl+C to stop.')
    print()

    # Counters for the summary printed on clean shutdown
    n_flows   = 0   # Flows sent to ML queue
    n_alerts  = 0   # Suricata signature alerts logged
    n_skipped = 0   # Events skipped (already-alerted flows, extraction failures)

    # File tracking state — persists across iterations
    inode    = None   # Last known inode number (None = first run)
    position = 0      # Current byte position in eve.json

    while True:   # Run forever
        try:
            # ── Check for log rotation ────────────────────────────────────────
            try:
                current_inode = eve_path.stat().st_ino   # Get current file inode
            except FileNotFoundError:
                # eve.json doesn't exist yet (Suricata not started?) — wait
                time.sleep(POLL_INTERVAL)
                continue

            if inode != current_inode:
                # Inode changed = new file (rotated) or first run
                if inode is not None:
                    log.info('eve.json rotated — reopening from start')
                inode    = current_inode
                position = 0   # Start reading from beginning of new file

            # ── Read new lines since last check ───────────────────────────────
            with open(eve_path, 'r') as f:
                f.seek(position)           # Jump to where we left off last time
                lines    = f.readlines()   # Read only new lines
                position = f.tell()        # Remember new end position

            if not lines:
                # No new events — sleep and check again
                time.sleep(POLL_INTERVAL)
                continue

            # Accumulate flow rows to write as one batch (one lock acquisition)
            batch = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue   # Skip blank lines

                # Parse the JSON line — skip if malformed
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get('event_type', '')

                # ── PATH 1: Suricata signature alert ──────────────────────────
                # Suricata matched this traffic against a known rule in its
                # Emerging Threats ruleset. We don't need the ML model for this —
                # just log it immediately with full confidence.
                if event_type == 'alert':
                    n_alerts += 1
                    alert     = event.get('alert', {})
                    sig       = alert.get('signature', 'Unknown')   # Rule name
                    category  = alert.get('category', '')
                    severity  = alert.get('severity', '')
                    src_ip    = event.get('src_ip',   '')
                    src_port  = event.get('src_port', '')
                    dest_ip   = event.get('dest_ip',  '')
                    dest_port = event.get('dest_port','')
                    proto     = event.get('proto',    '')
                    ts        = event.get('timestamp','')

                    msg = (f'KNOWN ATTACK  {src_ip}:{src_port} → '
                           f'{dest_ip}:{dest_port}  [{sig}]')
                    print(f'{RED}{msg}{RESET}')
                    log.info(msg)

                    # Write structured entry to alerts CSV
                    write_alert(alerts_path, {
                        'timestamp'       : ts,
                        'type'            : 'KNOWN_ATTACK',
                        'src_ip'          : src_ip,
                        'src_port'        : src_port,
                        'dest_ip'         : dest_ip,
                        'dest_port'       : dest_port,
                        'proto'           : proto,
                        'alert_signature' : sig,
                        'alert_category'  : category,
                        'alert_severity'  : severity,
                        'confidence'      : 1.0,   # Signature match = certain
                        'flow_id'         : event.get('flow_id', ''),
                    })

                    # Also notify syslog for any external monitoring tools
                    send_syslog(
                        syslog.LOG_WARNING,
                        f'KNOWN_ATTACK src={src_ip}:{src_port} '
                        f'dst={dest_ip}:{dest_port} sig="{sig}"'
                    )

                # ── PATH 2: Unalerted flow → ML queue ────────────────────────
                # Suricata completed tracking this flow but no signature matched.
                # This is the traffic our Random Forest model needs to classify.
                elif event_type == 'flow':
                    flow_info = event.get('flow', {})
                    alerted   = flow_info.get('alerted', False)

                    if alerted:
                        # This flow DID trigger an alert (handled in PATH 1).
                        # Skip it here to avoid double processing.
                        n_skipped += 1
                        continue

                    # Extract the 19 numerical features
                    features = extract_features(event)
                    if features is None:
                        # Extraction failed (malformed data) — skip this flow
                        n_skipped += 1
                        continue

                    # Build the complete queue row:
                    # metadata columns first, then the 19 feature columns
                    row = {
                        'flow_id'   : event.get('flow_id',   ''),
                        'timestamp' : event.get('timestamp', ''),
                        'src_ip'    : event.get('src_ip',    ''),
                        'src_port'  : event.get('src_port',  ''),
                        'dest_ip'   : event.get('dest_ip',   ''),
                        'dest_port' : event.get('dest_port', ''),
                        'proto'     : event.get('proto',     ''),
                    }
                    row.update(features)   # Merge the 19 feature values in
                    batch.append(row)
                    n_flows += 1

                # All other event types (dns, http, tls, stats…) are ignored

            # ── Write the accumulated batch to flows.csv ───────────────────────
            # One lock acquisition per loop iteration — much more efficient
            # than locking once per individual row
            if batch:
                queue_write(queue_path, batch)
                log.info(f'Queued {len(batch)} flow(s) '
                         f'[total: flows={n_flows} alerts={n_alerts} '
                         f'skipped={n_skipped}]')

        except KeyboardInterrupt:
            # Ctrl+C pressed — print summary and exit cleanly
            print()
            log.info('Extractor stopped.')
            log.info(f'  Total flows queued : {n_flows}')
            log.info(f'  Total alerts logged: {n_alerts}')
            log.info(f'  Total skipped      : {n_skipped}')
            sys.exit(0)

        except Exception as e:
            # Unexpected error — log it and keep running
            # (systemd will restart us if we crash, but better to stay up)
            log.error(f'Unexpected error: {e}')
            time.sleep(1)

        time.sleep(POLL_INTERVAL)   # Wait before checking for new events again


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IDS Extractor — extracts Suricata flow features to queue'
    )
    # All paths can be overridden on the command line for testing
    parser.add_argument('--eve',    default=DEFAULT_EVE,
                        help='Path to Suricata eve.json')
    parser.add_argument('--queue',  default=DEFAULT_QUEUE,
                        help='Path to queue CSV file')
    parser.add_argument('--alerts', default=DEFAULT_ALERTS,
                        help='Path to alerts log CSV')
    args = parser.parse_args()

    # Create output directories if they don't already exist
    Path(args.queue).parent.mkdir(parents=True, exist_ok=True)
    Path(args.alerts).parent.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  IDS Extractor — Service 1')
    print('  Suricata eve.json → Feature Queue')
    print(f'  Features: {len(FEATURE_NAMES)}')
    print('=' * 60)

    watch_eve(args.eve, args.queue, args.alerts)


if __name__ == '__main__':
    main()
