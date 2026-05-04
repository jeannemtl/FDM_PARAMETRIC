"""
Channel schema + reasoning template logic.

Mirrors Tables 12-14 of paper draft 18 (40-channel schema, n-hop templates,
worked decision logic from Appendix C). The reasoning logic is the source of
truth for what counts as a "correct answer" — used by both data generation
and evaluation.
"""

from typing import Dict, List, Tuple, Optional
import random


# ============================================================
# Channel schema (Table 12)
# ============================================================

CHANNEL_VOCAB: Dict[int, Tuple[str, List[str]]] = {
    0:  ("SECRET",     ["RED", "BLUE", "GREEN", "GOLD"]),
    1:  ("LOCATION",   ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2:  ("AGENT",      ["ALICE", "BOB", "CAROL", "DAVE"]),
    3:  ("STATUS",     ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4:  ("PRIORITY",   ["HIGH", "MEDIUM", "LOW"]),
    5:  ("BACKUP",     ["AVAILABLE", "UNAVAILABLE"]),
    6:  ("RULE",       ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7:  ("META",       ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY",
                         "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",       ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",     ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",      ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",       ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",      ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",     ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",      ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",    ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",     ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",    ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",    ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",    ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",     ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",       ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",    ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",      ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION",   ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",    ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",       ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE",   ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",      ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION",  ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",      ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",     ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",     ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",     ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",      ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",       ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2",   ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE",   ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",      ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}

CHANNEL_NAMES = {idx: name for idx, (name, _) in CHANNEL_VOCAB.items()}


def sample_channel_values(rng: random.Random) -> Dict[int, str]:
    """Sample one value per channel from its vocabulary."""
    return {idx: rng.choice(values) for idx, (_, values) in CHANNEL_VOCAB.items()}


# ============================================================
# Reasoning templates (Tables 13-14 + Appendix C decision logic)
# ============================================================

def reason_1hop_status(v): return v[3]
def reason_1hop_threat(v): return v[16]
def reason_1hop_rule(v): return v[6]
def reason_1hop_location(v): return v[1]


def reason_2hop_rule_status(v):
    """RULE -> STATUS (ch6, ch3)."""
    rule, status = v[6], v[3]
    if rule == "SAFETY_FIRST":
        return {"COMPROMISED": "ABORT", "CLEAR": "PROCEED", "UNKNOWN": "WAIT"}[status]
    if rule == "MISSION_FIRST":
        return "PROCEED_WITH_CAUTION" if status == "COMPROMISED" else "PROCEED"
    if rule == "BALANCED":
        return "ABORT" if status == "COMPROMISED" else "PROCEED"
    if rule == "CAUTIOUS":
        return "PROCEED" if status == "CLEAR" else "ABORT"


def reason_2hop_threat_terrain(v):
    """THREAT -> TERRAIN (ch16, ch18)."""
    threat, terrain = v[16], v[18]
    if threat in ("HIGH", "CRITICAL"):
        return "EVACUATION"
    if threat == "MEDIUM":
        return "FORTIFY" if terrain in ("URBAN", "COASTAL") else "CONTINUE"
    return "CONTINUE"


def reason_2hop_agent_location(v):
    """AGENT -> LOCATION (ch2, ch1) — pure read, no branching."""
    return f"{v[2]}_AT_{v[1]}"


def reason_3hop_rule_status_backup(v):
    """RULE -> STATUS -> BACKUP (ch6, ch3, ch5)."""
    rule, status, backup = v[6], v[3], v[5]
    if rule == "CAUTIOUS":
        return "PROCEED" if (status == "CLEAR" and backup == "AVAILABLE") else "ABORT"
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return "ABORT"
        return "PROCEED" if backup == "AVAILABLE" else "HOLD"
    if rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            return "ABORT"
        return "PROCEED"
    if rule == "BALANCED":
        score = (1 if status == "CLEAR" else 0) + (1 if backup == "AVAILABLE" else 0)
        return {2: "PROCEED", 1: "CAUTION", 0: "ABORT"}[score]


def reason_3hop_threat_cover_signal(v):
    """THREAT -> COVER -> SIGNAL (ch16, ch14, ch32)."""
    threat, cover, signal = v[16], v[14], v[32]
    if threat in ("HIGH", "CRITICAL") and cover == "NONE":
        return "CRITICAL_EXPOSURE"
    if signal in ("JAMMED", "LOST"):
        return "COMMS_COMPROMISED"
    return "MANAGEABLE"


def reason_4hop_meta_rule_status_priority(v):
    """META -> RULE -> STATUS -> PRIORITY (ch7, ch6, ch3, ch4)."""
    meta, rule, status, priority = v[7], v[6], v[3], v[4]
    if meta == "EMERGENCY":
        return "EMERGENCY_RESPONSE"
    if meta == "LOCKDOWN":
        return "LOCKDOWN_ACTIVE"
    if meta == "OVERRIDE_STATUS":
        return "PROCEED" if priority == "HIGH" else "CAUTION"
    # NONE or OVERRIDE_PRIORITY: full chain
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        return "ABORT"
    return "PROCEED" if priority == "HIGH" else "HOLD"


def reason_4hop_threat_terrain_weather_asset(v):
    """THREAT -> TERRAIN -> WEATHER -> ASSET (ch16, ch18, ch17, ch12)."""
    threat, terrain, weather, asset = v[16], v[18], v[17], v[12]
    if threat in ("HIGH", "CRITICAL") and weather == "STORM":
        return "GROUND"
    if asset in ("AIRCRAFT", "DRONE") and weather == "FOG":
        return "DELAY"
    if terrain == "MOUNTAIN" and asset == "BOAT":
        return "REASSIGN"
    return "DEPLOY"


def reason_5hop_meta_rule_status_backup_threat(v):
    """META -> RULE -> STATUS -> BACKUP -> THREAT (ch7, ch6, ch3, ch5, ch16)."""
    meta, rule, status, backup, threat = v[7], v[6], v[3], v[5], v[16]
    if meta == "EMERGENCY":
        return "PROCEED"
    if meta == "LOCKDOWN":
        return "ABORT"
    # full chain
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        return "EMERGENCY_EXTRACT" if threat in ("HIGH", "CRITICAL") else "STANDARD_ABORT"
    if backup == "UNAVAILABLE" and threat in ("HIGH", "CRITICAL"):
        return "ABORT_WITH_EXTRACT"
    if backup == "AVAILABLE":
        return "PROCEED_WITH_BACKUP"
    return "HOLD"


def reason_5hop_weather_visibility_window_terrain_extract(v):
    """WEATHER -> VISIBILITY -> WINDOW -> TERRAIN -> EXTRACT
    (ch17, ch28, ch13, ch18, ch19)."""
    weather, vis, window, terrain, extract = v[17], v[28], v[13], v[18], v[19]
    if weather == "STORM" and vis == "ZERO":
        return "EVACUATE" if extract == "READY" else "SHELTER"
    if window == "NIGHT" and vis == "REDUCED":
        return "HOLD"
    if terrain == "MOUNTAIN" and weather in ("STORM", "FOG"):
        return "DESCEND"
    return "CONTINUE"


# Template registry: (hop_count, name, channels_used, question, reasoner)
TEMPLATES: List[Tuple[int, str, List[int], str, callable]] = [
    # 1-hop
    (1, "1h_status",   [3],  "Report current status.",     reason_1hop_status),
    (1, "1h_threat",   [16], "Report threat level.",       reason_1hop_threat),
    (1, "1h_rule",     [6],  "Report active rule.",        reason_1hop_rule),
    (1, "1h_location", [1],  "Report current location.",   reason_1hop_location),
    # 2-hop
    (2, "2h_rule_status",     [6, 3],  "Evaluate status under current rule.", reason_2hop_rule_status),
    (2, "2h_threat_terrain",  [16, 18],"Assess threat in current terrain.",   reason_2hop_threat_terrain),
    (2, "2h_agent_location",  [2, 1],  "Report agent deployment location.",   reason_2hop_agent_location),
    # 3-hop
    (3, "3h_rule_status_backup",  [6, 3, 5],
        "Should we proceed given rule, status, and backup?", reason_3hop_rule_status_backup),
    (3, "3h_threat_cover_signal", [16, 14, 32],
        "Evaluate threat exposure with cover and signal status.", reason_3hop_threat_cover_signal),
    # 4-hop
    (4, "4h_meta_rule_status_priority", [7, 6, 3, 4],
        "Full override check: meta, rule, status, priority.", reason_4hop_meta_rule_status_priority),
    (4, "4h_threat_terrain_weather_asset", [16, 18, 17, 12],
        "Tactical assessment: threat, terrain, weather, asset.", reason_4hop_threat_terrain_weather_asset),
    # 5-hop
    (5, "5h_meta_rule_status_backup_threat", [7, 6, 3, 5, 16],
        "Complete chain: meta override, rule, status, backup, threat.",
        reason_5hop_meta_rule_status_backup_threat),
    (5, "5h_weather_visibility_window_terrain_extract", [17, 28, 13, 18, 19],
        "Environmental chain: weather, visibility, window, terrain, extract.",
        reason_5hop_weather_visibility_window_terrain_extract),
]


def get_template(hop_count: int, rng: random.Random) -> Tuple[str, List[int], str, callable]:
    """Sample one template at the requested hop count."""
    candidates = [(name, chs, q, fn) for h, name, chs, q, fn in TEMPLATES if h == hop_count]
    return rng.choice(candidates)
