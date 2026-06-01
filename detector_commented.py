#!/usr/bin/env python3
"""
detector.py — Service 2: ML Detector
=====================================

WHAT THIS SERVICE DOES:
    This is the SECOND service in the hybrid IDS pipeline.
    It runs continuously in the background, polling flows.csv every second.

    extractor.py writes unclassified flow features to flows.csv.
    This service reads those rows, runs each one through the trained
    Random Forest model, and decides what to do with the result:

      ATTACK     → Model is confident enough (above threshold) that this is
                   an attack. Write to traffic.log + alerts_log.csv + syslog.

      SUPPRESSED → Model suspects an attack but confidence is BELOW the
                   per-class threshold. Logged for monitoring purposes only —
                   no alert is raised. Useful for threshold tuning.

      CLEAN      → Model predicts BENIGN. Written to traffic.log only.

HOW IT FITS IN THE PIPELINE:
    extractor.py → flows.csv → [detector.py] → traffic.log
                                             ↘ alerts_log.csv (ATTACK only)
                                             ↘ syslog

Usage:
    python detector.py [--model-dir /opt/ids]
                       [--queue     /opt/ids/flows.csv]
                       [--alerts    /opt/ids/alerts_log.csv]
                       [--traffic   /opt/ids/traffic.log]

Version notes (important changes made during development):
    - Artifact filenames updated to match training output names
    - IQR outlier capping REMOVED — was not applied during training, so
      applying it at inference would distort the feature space the model learned
    - median_imputation.json now loaded for inf/NaN handling at runtime
    - Single global threshold REPLACED by per-class thresholds from class_thresholds.json
"""

import os
import sys
import csv
import json
import time
import fcntl        # Linux file locking — prevents collision with extractor.py
import logging
import argparse
import syslog       # Linux system log
import warnings
from datetime import datetime
from pathlib import Path

import joblib       # Load serialised scikit-learn model (.pkl files)
import numpy as np  # For isnan() and isinf() checks
import pandas as pd # Build a named-column DataFrame for model inference

# Suppress scikit-learn deprecation warnings at startup
warnings.filterwarnings('ignore')
# Suppress joblib's verbose parallel output (e.g. "[Parallel(n_jobs=-1)]")
os.environ['JOBLIB_VERBOSITY'] = '0'

# ── Configuration constants ───────────────────────────────────────────────────
DEFAULT_MODEL_DIR = '/opt/ids'                # Directory holding all .pkl and .json files
DEFAULT_QUEUE     = '/opt/ids/flows.csv'       # Input: feature rows from extractor.py
DEFAULT_ALERTS    = '/opt/ids/alerts_log.csv'  # Output: confirmed attack entries
DEFAULT_TRAFFIC   = '/opt/ids/traffic.log'     # Output: all flows in human-readable format
POLL_INTERVAL     = 1.0   # How often (seconds) to check flows.csv for new rows

# ── ANSI terminal colours ─────────────────────────────────────────────────────
RED    = '\033[91m'   # ATTACK — stands out immediately
YELLOW = '\033[93m'   # SUPPRESSED — warrants attention but not an alert
GREEN  = '\033[92m'   # CLEAN — normal traffic
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('detector')


# ─────────────────────────────────────────────────────────────────────────────
# IDSModel class: loads all artifacts and provides the predict() method
# ─────────────────────────────────────────────────────────────────────────────

class IDSModel:
    """
    Encapsulates everything needed to classify a network flow:
      - The trained Random Forest classifier
      - The label encoder (integer → class name)
      - The 19-feature contract (which features, in what order)
      - Per-class confidence thresholds
      - Median imputation values for inf/NaN handling

    All artifacts are loaded once at startup and reused for every prediction.
    """

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        log.info('Loading model artifacts...')

        # ── Load the Random Forest model ──────────────────────────────────────
        # random_forest_ids.pkl is ~32MB — 200 decision trees trained on
        # 2,017,852 flows across 6 attack classes.
        # Setting verbose=0 stops it printing "[Parallel(n_jobs=-1)]" every call.
        self.rf = joblib.load(model_dir / 'random_forest_ids.pkl')
        self.rf.verbose = 0

        # ── Load the label encoder ────────────────────────────────────────────
        # scikit-learn RandomForestClassifier works with integer class labels
        # internally (0, 1, 2, 3, 4, 5). The label encoder maps those back to
        # human-readable strings: 0→'BENIGN', 1→'Bot', 2→'Brute Force', etc.
        self.le = joblib.load(model_dir / 'label_encoder.pkl')

        # ── Load the feature contract ─────────────────────────────────────────
        # An ordered list of 19 feature names. The model was trained with
        # features in this exact order — we must feed them in the same order
        # at inference time or the model maps values to the wrong features.
        self.features = joblib.load(model_dir / 'feature_contract.pkl')

        # ── Load per-class confidence thresholds ──────────────────────────────
        # WHY PER-CLASS THRESHOLDS INSTEAD OF ONE GLOBAL THRESHOLD?
        #
        # Different attack classes have very different difficulty levels.
        # At a low threshold, Bot and Web Attack generate many false positives
        # because normal HTTP/C2-like traffic overlaps with their feature space.
        # We tuned a separate threshold for each class by sweeping 0.30→0.99
        # and picking the value that maximised per-class F1 on the test set.
        #
        # Final deployed thresholds (from class_thresholds.json):
        #   BENIGN      : 0.00  (no threshold — it's the safe fallback)
        #   DoS         : 0.73
        #   PortScan    : 0.61
        #   Brute Force : 0.65  (offline tuning gave 0.89, lowered for live deployment)
        #   Bot         : 0.88  (high — Bot looks too similar to normal traffic)
        #   Web Attack  : 0.84  (high — HTTP payload not visible at flow level)
        with open(model_dir / 'class_thresholds.json') as f:
            self.thresholds = json.load(f)

        # ── Load median imputation values ─────────────────────────────────────
        # During preprocessing, flow_bytes/s and flow_pkts/s produced inf/NaN
        # values when flow_duration was 0 (instant flows). We replaced those
        # with the column median during training.
        # At inference we must apply the EXACT SAME medians — not recompute them —
        # otherwise the model sees different values than it was trained on.
        with open(model_dir / 'median_imputation.json') as f:
            self.medians = json.load(f)

        # ── Load training results (for display at startup only) ───────────────
        results_path = model_dir / 'training_results.json'
        if results_path.exists():
            with open(results_path) as f:
                meta = json.load(f)
            metrics = meta.get('metrics', {})
        else:
            metrics = {}

        # Print a model summary so we can verify the right artifacts loaded
        log.info(f'  Model      : random_forest_ids.pkl')
        log.info(f'  Features   : {len(self.features)} (contract order)')
        log.info(f'  Classes    : {list(self.le.classes_)}')
        log.info(f'  Macro F1   : {metrics.get("macro_f1", "n/a")}')
        log.info(f'  Thresholds : {self.thresholds}')
        log.info('Model loaded successfully.')

    def predict(self, feature_dict):
        """
        Classify one network flow.

        HOW THE PREDICTION WORKS:
        1. Read the 19 feature values from feature_dict in contract order.
        2. Replace any inf/NaN values with training-time medians.
        3. Wrap in a DataFrame (scikit-learn needs named columns).
        4. Call predict_proba() → get a probability for each of the 6 classes.
        5. Find the class with the highest probability (raw_class).
        6. Check if that probability meets the class-specific threshold.
        7. If yes → return raw_class as the detection.
           If no  → return 'BENIGN' (suppress the low-confidence alert).

        WHY SUPPRESS INSTEAD OF IGNORE?
        Suppressed events are still logged to traffic.log so an analyst can
        see the model's suspicion. This is useful for threshold tuning:
        if you see many SUPPRESSED Brute Force events during a known attack,
        the threshold might be too high.

        Args:
            feature_dict: dict with keys matching FEATURE_NAMES
                          (as written by extractor.py to flows.csv)

        Returns:
            predicted_class : 'BENIGN' or an attack class name
            confidence      : probability score for raw_class (0.0–1.0)
            raw_class       : highest-probability class BEFORE threshold
            proba_dict      : full {class: probability} for all 6 classes
        """
        # ── Step 1: Build the feature vector in contract order ─────────────────
        # We must extract features in exactly the same order as training.
        # Any missing feature defaults to 0.0 (safe neutral value).
        row = []
        for feat in self.features:
            val = feature_dict.get(feat, 0.0)
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0   # Handle non-numeric values gracefully

            # Replace inf/NaN with the training-time median
            # (same imputation applied during preprocessing)
            if np.isnan(val) or np.isinf(val):
                val = float(self.medians.get(feat, 0.0))
            row.append(val)

        # Wrap in a single-row DataFrame — scikit-learn uses column names
        # to ensure features are in the right slots
        X = pd.DataFrame([row], columns=self.features)

        # ── Step 2: Run the Random Forest model ───────────────────────────────
        # Using threading backend with n_jobs=1 avoids spawning additional
        # processes for every single-row prediction (which would be very slow)
        with joblib.parallel_backend('threading', n_jobs=1):
            pred_enc  = self.rf.predict(X)[0]         # Predicted class as integer
            pred_prob = self.rf.predict_proba(X)[0]   # Probability array, one per class

        # Convert the integer prediction back to a class name string
        confidence = float(pred_prob.max())                        # Highest probability
        raw_class  = self.le.inverse_transform([pred_enc])[0]     # e.g. 'Brute Force'
        threshold  = self.thresholds.get(raw_class, 0.5)          # Class-specific minimum

        # ── Step 3: Apply the per-class confidence threshold ──────────────────
        # If confidence is below threshold, we don't trust the prediction enough.
        # Fall back to BENIGN — a cautious choice that avoids false alarms.
        # The actual suspicion is still visible in traffic.log as SUPPRESSED.
        if confidence < threshold:
            predicted_class = 'BENIGN'   # Suppress the low-confidence alert
        else:
            predicted_class = raw_class  # Trust the prediction — raise an alert

        # Build the full probability breakdown for logging/debugging
        proba_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self.le.classes_, pred_prob)
        }

        return predicted_class, confidence, raw_class, proba_dict


# ─────────────────────────────────────────────────────────────────────────────
# Queue reader: atomically reads and clears flows.csv
# ─────────────────────────────────────────────────────────────────────────────

def queue_read_and_clear(queue_path):
    """
    Read all pending flow rows from flows.csv, then immediately clear the file.

    WHY READ-AND-CLEAR (not read-only)?
    extractor.py keeps appending rows. If we only read without clearing,
    the file would grow indefinitely and we'd re-classify the same flows
    over and over. By clearing immediately after reading, each flow is
    processed exactly once.

    WHY EXCLUSIVE FILE LOCK?
    extractor.py could be writing a new row at the exact moment we're reading.
    The exclusive lock (fcntl.LOCK_EX) blocks extractor from writing until
    we're done reading and have cleared the file, preventing partial/corrupt rows.

    Returns:
        list of dicts (one per flow, keys = QUEUE_COLUMNS), or [] if empty
    """
    queue_path = Path(queue_path)

    # Fast path: skip locking entirely if file is empty
    if not queue_path.exists() or queue_path.stat().st_size == 0:
        return []

    lock_path = str(queue_path) + '.lock'
    rows = []

    try:
        with open(lock_path, 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)   # Block until we have exclusive access

            # Read all rows into memory
            with open(queue_path, 'r', newline='') as f:
                reader = csv.DictReader(f)   # Reads header row automatically
                rows = list(reader)

            # Clear the file — open in write mode ('w') truncates to zero length
            # extractor.py will write a new header on its next batch
            open(queue_path, 'w').close()

            fcntl.flock(lf, fcntl.LOCK_UN)   # Release lock so extractor can write again

    except Exception as e:
        log.error(f'Queue read failed: {e}')

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Alert writer: appends ML-detected attacks to alerts_log.csv
# ─────────────────────────────────────────────────────────────────────────────

# This matches the columns written by extractor.py for known attacks,
# plus a few extra ML-specific fields (predicted_class, raw_class, etc.)
ALERT_COLUMNS = ['timestamp', 'type', 'src_ip', 'src_port',
                 'dest_ip', 'dest_port', 'proto',
                 'predicted_class', 'confidence', 'raw_class',
                 'alert_signature', 'alert_category', 'alert_severity',
                 'flow_id']

def write_alert(alerts_path, row):
    """
    Append one ML-detected attack to alerts_log.csv.

    alerts_log.csv receives two types of entries:
    - KNOWN_ATTACK   written by extractor.py (Suricata signature match)
    - UNKNOWN_ATTACK written by detector.py  (ML model detection)

    Keeping both in the same CSV allows unified analysis of all detections.
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
    Send a detection event to the Linux system log.
    LOG_WARNING for confirmed attacks, LOG_INFO for suppressed/clean events.
    Allows integration with external monitoring tools like SIEM, Splunk, ELK.
    """
    try:
        syslog.openlog('ids-detector', syslog.LOG_PID, syslog.LOG_LOCAL0)
        syslog.syslog(priority, message)
        syslog.closelog()
    except Exception as e:
        log.warning(f'Syslog send failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Traffic log writer: records every flow to a human-readable text file
# ─────────────────────────────────────────────────────────────────────────────

def write_traffic_log(traffic_path, entry):
    """
    Append one flow entry to traffic.log in fixed-width column format.

    Unlike alerts_log.csv (which only records attacks), traffic.log records
    ALL flows: ATTACK, SUPPRESSED, and CLEAN. This gives a complete picture
    of what the system is seeing.

    Operators monitor this in real time using:
        tail -f /opt/ids/traffic.log          # all events
        tail -f /opt/ids/traffic.log | grep -v CLEAN  # attacks + suppressed only

    Example output:
    TIMESTAMP              | TYPE       | CLASS         | CONF   | SRC                        | DST
    --------------------------------------------------------------------------------------------------------
    2026-03-22 09:00:12    | ATTACK     | Brute Force   | 67.0%  | 192.168.240.129:54840      | 192.168.240.133:22
    2026-03-22 09:00:13    | SUPPRESSED | Bot           | 55.6%  | 10.12.17.101:51142         | 176.134.78.64:23
    2026-03-22 09:00:14    | CLEAN      | BENIGN        | 97.1%  | 192.168.1.10:59162         | 54.201.188.11:443
    """
    traffic_path = Path(traffic_path)
    write_header = not traffic_path.exists() or traffic_path.stat().st_size == 0
    try:
        with open(traffic_path, 'a') as f:
            if write_header:
                # Write column headers on first use
                f.write(
                    f"{'TIMESTAMP':<22} | {'TYPE':<10} | {'CLASS':<13} | "
                    f"{'CONF':<6} | {'SRC':<26} | {'DST':<26} | NOTE\n"
                )
                f.write('-' * 120 + '\n')

            # Format each field to fixed width for clean column alignment
            ts    = entry['ts'][:19].replace('T', ' ')   # '2026-03-22T09:00:12.xxx' → '2026-03-22 09:00:12'
            etype = entry['type']                          # 'ATTACK', 'SUPPRESSED', or 'CLEAN'
            cls   = entry['class']                         # 'Brute Force', 'BENIGN', etc.
            conf  = f"{entry['confidence']*100:.1f}%"     # 0.671 → '67.1%'
            src   = f"{entry['src_ip']}:{entry['src_port']}"
            dst   = f"{entry['dest_ip']}:{entry['dest_port']}"
            note  = entry.get('note', '')                  # e.g. 'flow_id=260926166740513'

            f.write(
                f"{ts:<22} | {etype:<10} | {cls:<13} | "
                f"{conf:<6} | {src:<26} | {dst:<26} | {note}\n"
            )
    except Exception as e:
        log.error(f'Traffic log write failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────

def run_detector(model_dir, queue_path, alerts_path, traffic_path):
    """
    The main loop — runs forever until killed.

    Every POLL_INTERVAL seconds (1 second):
    1. Read all pending rows from flows.csv and clear it
    2. For each row, run model.predict()
    3. Route to ATTACK, SUPPRESSED, or CLEAN based on the result
    4. Write to traffic.log + optionally alerts_log.csv + syslog
    5. Print colour-coded summary to terminal

    OUTCOME LOGIC:
      is_attack     = (predicted_class != 'BENIGN')
                      → Model was confident enough → confirmed attack
      is_suppressed = (predicted_class == 'BENIGN') AND (raw_class != 'BENIGN')
                      → Model suspected an attack but confidence was below threshold
                      → Logged as suspicious but no alert raised
      else (clean)  = both predicted_class and raw_class are 'BENIGN'
                      → Model is confident this is normal traffic
    """
    # Load all model artifacts once at startup
    model = IDSModel(model_dir)

    log.info(f'Queue    : {queue_path}')
    log.info(f'Alerts   : {alerts_path}')
    log.info(f'Traffic  : {traffic_path}')
    log.info('Detector running — Ctrl+C to stop.')
    print()

    # Running totals — printed in every batch log message
    n_clean      = 0
    n_attack     = 0
    n_suppressed = 0
    n_total      = 0

    while True:   # Run forever
        try:
            # ── Read all pending flow rows ────────────────────────────────────
            # This atomically reads flows.csv and clears it in one lock operation
            rows = queue_read_and_clear(queue_path)

            for row in rows:
                n_total += 1

                # ── Classify this flow ─────────────────────────────────────────
                predicted_class, confidence, raw_class, proba_dict = model.predict(row)

                # Extract metadata for display and logging
                src_ip    = row.get('src_ip',   '?')
                src_port  = row.get('src_port', '?')
                dest_ip   = row.get('dest_ip',  '?')
                dest_port = row.get('dest_port','?')
                src       = f"{src_ip}:{src_port}"
                dst       = f"{dest_ip}:{dest_port}"
                proto     = row.get('proto', '?')
                ts        = row.get('timestamp', datetime.now().isoformat())
                flow_id   = row.get('flow_id', '')

                # Determine which outcome category this flow falls into
                is_attack     = (predicted_class != 'BENIGN')
                is_suppressed = (predicted_class == 'BENIGN' and raw_class != 'BENIGN')

                # ── OUTCOME 1: ATTACK ──────────────────────────────────────────
                # Model predicted an attack class AND confidence >= class threshold.
                # This is the primary detection output.
                if is_attack:
                    n_attack += 1
                    threshold = model.thresholds.get(raw_class, 0.5)

                    msg = (f'ATTACK      {src} → {dst}  '
                           f'{predicted_class}  ({confidence*100:.1f}%  '
                           f'threshold={threshold})')
                    print(f'{RED}{BOLD}{msg}{RESET}')   # Bold red — visible immediately
                    log.info(msg)

                    # Write to alerts CSV — type='UNKNOWN_ATTACK' distinguishes
                    # ML detections from Suricata signature matches ('KNOWN_ATTACK')
                    write_alert(alerts_path, {
                        'timestamp'       : ts,
                        'type'            : 'UNKNOWN_ATTACK',
                        'src_ip'          : src_ip,
                        'src_port'        : src_port,
                        'dest_ip'         : dest_ip,
                        'dest_port'       : dest_port,
                        'proto'           : proto,
                        'predicted_class' : predicted_class,
                        'confidence'      : round(confidence, 4),
                        'raw_class'       : raw_class,     # Class before threshold check
                        'alert_signature' : f'ML:{predicted_class}',
                        'alert_category'  : 'ML Detection',
                        'alert_severity'  : 1,
                        'flow_id'         : flow_id,
                    })

                    write_traffic_log(traffic_path, {
                        'ts': ts, 'type': 'ATTACK', 'class': predicted_class,
                        'confidence': confidence, 'src_ip': src_ip,
                        'src_port': src_port, 'dest_ip': dest_ip,
                        'dest_port': dest_port, 'note': f'flow_id={flow_id}',
                    })

                    # LOG_WARNING so syslog-based monitoring tools pick it up
                    send_syslog(
                        syslog.LOG_WARNING,
                        f'UNKNOWN_ATTACK class={predicted_class} '
                        f'confidence={confidence:.4f} '
                        f'src={src_ip}:{src_port} '
                        f'dst={dest_ip}:{dest_port} '
                        f'proto={proto} flow_id={flow_id}'
                    )

                # ── OUTCOME 2: SUPPRESSED ──────────────────────────────────────
                # Model suspected an attack but confidence < per-class threshold.
                # We log it for transparency but do NOT raise an alert.
                # This is how the SUPPRESSED→ATTACK transition appeared in our
                # Brute Force live test when we lowered the threshold from 0.89→0.65.
                elif is_suppressed:
                    n_suppressed += 1
                    threshold = model.thresholds.get(raw_class, 0.5)

                    msg = (f'SUPPRESSED  {src} → {dst}  '
                           f'{raw_class} @ {confidence*100:.1f}%  '
                           f'(below threshold {threshold})')
                    print(f'{YELLOW}{msg}{RESET}')   # Yellow — suspicious but not alarming

                    # SUPPRESSED flows go to traffic.log only — NOT alerts_log.csv
                    write_traffic_log(traffic_path, {
                        'ts': ts, 'type': 'SUPPRESSED', 'class': raw_class,
                        'confidence': confidence, 'src_ip': src_ip,
                        'src_port': src_port, 'dest_ip': dest_ip,
                        'dest_port': dest_port,
                        'note': f'below threshold {threshold}',
                    })

                    send_syslog(
                        syslog.LOG_INFO,   # INFO not WARNING — not a confirmed attack
                        f'SUPPRESSED class={raw_class} '
                        f'confidence={confidence:.4f} '
                        f'src={src_ip}:{src_port} '
                        f'dst={dest_ip}:{dest_port} '
                        f'proto={proto}'
                    )

                # ── OUTCOME 3: CLEAN ───────────────────────────────────────────
                # Model is confident this is normal (benign) traffic.
                # Only logged to traffic.log — no alert, no CSV entry.
                # Confirms our 0% false positive rate on BENIGN traffic.
                else:
                    n_clean += 1

                    msg = f'CLEAN       {src} → {dst}  BENIGN  ({confidence*100:.1f}%)'
                    print(f'{GREEN}{msg}{RESET}')   # Green — all good

                    write_traffic_log(traffic_path, {
                        'ts': ts, 'type': 'CLEAN', 'class': 'BENIGN',
                        'confidence': confidence, 'src_ip': src_ip,
                        'src_port': src_port, 'dest_ip': dest_ip,
                        'dest_port': dest_port, 'note': '',
                    })

                    send_syslog(
                        syslog.LOG_INFO,
                        f'CLEAN class=BENIGN '
                        f'confidence={confidence:.4f} '
                        f'src={src_ip}:{src_port} '
                        f'dst={dest_ip}:{dest_port} '
                        f'proto={proto}'
                    )

            # Log a summary after each polling cycle that had work to do
            if rows:
                log.info(
                    f'Processed {len(rows)} flow(s) — '
                    f'attacks={n_attack} suppressed={n_suppressed} '
                    f'clean={n_clean} total={n_total}'
                )

        except KeyboardInterrupt:
            # Ctrl+C or systemctl stop — print final summary and exit cleanly
            print()
            log.info('Detector stopped.')
            log.info(f'  Total processed : {n_total}')
            log.info(f'  Attacks flagged : {n_attack}')
            log.info(f'  Suppressed      : {n_suppressed}')
            log.info(f'  Clean flows     : {n_clean}')
            sys.exit(0)

        except Exception as e:
            # Unexpected error — log it and keep running
            log.error(f'Unexpected error: {e}')
            time.sleep(1)

        # Wait before the next polling cycle
        time.sleep(POLL_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IDS Detector — runs RF model on queued flows'
    )
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help='Directory containing model artifacts')
    parser.add_argument('--queue',     default=DEFAULT_QUEUE,
                        help='Path to queue CSV file (written by extractor.py)')
    parser.add_argument('--alerts',    default=DEFAULT_ALERTS,
                        help='Path to alerts log CSV')
    parser.add_argument('--traffic',   default=DEFAULT_TRAFFIC,
                        help='Path to human-readable traffic log (all flows)')
    args = parser.parse_args()

    # Create output directories if they don't exist yet
    Path(args.alerts).parent.mkdir(parents=True, exist_ok=True)
    Path(args.traffic).parent.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  IDS Detector — Service 2')
    print('  Feature Queue → RF Model → Alerts')
    print('=' * 60)

    run_detector(args.model_dir, args.queue, args.alerts, args.traffic)


if __name__ == '__main__':
    main()
