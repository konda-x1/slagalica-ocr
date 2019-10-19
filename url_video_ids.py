import sys
import urllib.parse as urlparse
from urllib.parse import parse_qs

prefix = sys.argv[1]
urls = sys.argv[2:]
ids = []
for url in urls:
    parsed = parse_qs(urlparse.urlparse(url).query)['v'][0]
    ids.append(prefix + parsed)
print(*ids)
sys.exit(0)
