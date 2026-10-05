#!/usr/bin/env python3
"""Play a LeBron James video list in YouTube on a Roku (ECP).

Finds the Roku (SSDP, then a /24 scan of port 8060, or --ip), deep-links
the YouTube channel (app 837) to each video in turn, and moves on when
the Roku reports the player has stopped. YouTube's own autoplay chains
related videos if the script is not left running.

Needs the Roku on the network with Settings > System > Advanced system
settings > Control by mobile apps > Network access = Default/Permissive.

usage:
  python3 tools/roku_lebron.py [--ip 192.168.1.50] [--once] [--list FILE]
"""
import argparse
import re
import socket
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

YOUTUBE_APP_ID = '837'

# Commentary-heavy LeBron footage (play-by-play says the name often).
# Picked from YouTube search; the name-mute replay test used the first one.
LEBRON_VIDEOS = [
    ('W-KQ8rG2DRU', "LeBron James: The King's best Olympic moments | NBC Sports"),
    ('2nC9z57MuaI', "LeBron James' high school team upsets No. 1 Oak Hill Academy (2002) | ESPN Archive"),
    ('fnghnofk_y8', 'LeBron and AD combine for 58 points in Lakers vs. Clippers | 2019-20 NBA Highlights'),
    ('koplcs7BMIc', 'LeBron James EPIC MOMENTS From The 2018 Playoffs!'),
    ('LDw0gfoJGPU', 'NBA Legends on The Day Lebron James Ruthlessly DESTROYED The Boston Celtics'),
    ('7a6gnRvQqHQ', "The World's GREATEST LeBron James Highlight Reel"),
    ('FzeIZLOrHic', 'LEBRON JAMES HYPED PLAYS (LOUDEST CROWD REACTIONS EVER)'),
    ('EfWlOmGEfvU', 'LEBRON JAMES HYPED PLAYS (LOUDEST CROWD REACTIONS)'),
]


def ssdp_discover(timeout=4.0):
    msg = ('M-SEARCH * HTTP/1.1\r\nHost: 239.255.255.250:1900\r\n'
           'Man: "ssdp:discover"\r\nST: roku:ecp\r\nMX: 3\r\n\r\n')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.settimeout(timeout)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    for _ in range(3):
        s.sendto(msg.encode(), ('239.255.255.250', 1900))
    try:
        while True:
            data, _ = s.recvfrom(4096)
            m = re.search(r'(?i)location:\s*http://([\d.]+):8060', data.decode(errors='ignore'))
            if m:
                return m.group(1)
    except socket.timeout:
        return None


def scan_subnet():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        base = s.getsockname()[0].rsplit('.', 1)[0]
    finally:
        s.close()

    def probe(i):
        ip = f'{base}.{i}'
        c = socket.socket()
        c.settimeout(0.6)
        try:
            c.connect((ip, 8060))
            return ip
        except OSError:
            return None
        finally:
            c.close()

    with ThreadPoolExecutor(64) as ex:
        for ip in ex.map(probe, range(1, 255)):
            if ip:
                return ip
    return None


def ecp(ip, path, method='GET', timeout=5):
    req = urllib.request.Request(f'http://{ip}:8060{path}', method=method,
                                 data=b'' if method == 'POST' else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors='ignore')


def player_state(ip):
    try:
        xml = ecp(ip, '/query/media-player')
    except OSError:
        return None
    m = re.search(r'<player [^>]*state="([^"]+)"', xml)
    return m.group(1) if m else None


def play(ip, video_id):
    ecp(ip, f'/launch/{YOUTUBE_APP_ID}?contentId={video_id}&mediaType=movie', method='POST')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip')
    ap.add_argument('--once', action='store_true', help='start the first video and exit')
    ap.add_argument('--list', help='file with one YouTube video id per line')
    args = ap.parse_args()

    videos = LEBRON_VIDEOS
    if args.list:
        videos = [(ln.split()[0], ln.strip()) for ln in open(args.list) if ln.strip()]

    ip = args.ip or ssdp_discover() or scan_subnet()
    if not ip:
        print('No Roku found (is it on the network? Settings > Network)', file=sys.stderr)
        return 1
    info = ecp(ip, '/query/device-info')
    name = re.search(r'<user-device-name>(.*?)</', info) or re.search(r'<model-name>(.*?)</', info)
    print(f'Roku at {ip}: {name.group(1) if name else "?"}')

    i = 0
    while True:
        vid, title = videos[i % len(videos)]
        print(f'Playing {vid}: {title}', flush=True)
        play(ip, vid)
        if args.once:
            return 0
        time.sleep(20)  # let it start
        # Advance when the player stops (video ended or user backed out).
        idle = 0
        while idle < 3:
            st = player_state(ip)
            idle = idle + 1 if st in ('close', 'stop', 'none', None) else 0
            time.sleep(10)
        i += 1


if __name__ == '__main__':
    sys.exit(main())
