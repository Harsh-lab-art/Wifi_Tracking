import random, math
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AccessPoint:
    ssid: str
    bssid: str
    rssi: int
    channel: int = 0
    frequency: float = 2.4

def scan_wifi():
    return []

class RSSISimulator:
    _PROFILES = [
        {"ssid":"HomeNet_5G",  "bssid":"AA:BB:CC:11:22:33","base":-45,"chan":36},
        {"ssid":"Neighbor_2G","bssid":"AA:BB:CC:44:55:66","base":-65,"chan":6},
        {"ssid":"OfficeRouter","bssid":"AA:BB:CC:77:88:99","base":-72,"chan":11},
        {"ssid":"GuestNet",   "bssid":"AA:BB:CC:AA:BB:CC","base":-80,"chan":1},
    ]
    def __init__(self, person_present=False, seed=None):
        self.person_present = person_present
        self._rng = random.Random(seed)
        self._state = {p["bssid"]: float(p["base"]) for p in self._PROFILES}
        self._t = 0
    def scan(self):
        self._t += 1
        aps = []
        for p in self._PROFILES:
            b = p["bssid"]
            self._state[b] += 0.15*(p["base"]-self._state[b]) + self._rng.gauss(0,1.8)
            rssi = self._state[b]
            if self.person_present:
                if self._rng.random() < 0.35:
                    rssi += self._rng.uniform(-8,8)
                rssi += 1.5*math.sin(2*math.pi*self._t/12 + self._rng.uniform(0,6.28))
            aps.append(AccessPoint(ssid=p["ssid"],bssid=b,rssi=int(round(rssi)),
                                   channel=p["chan"],frequency=5.0 if p["chan"]>14 else 2.4))
        return aps