import sys
import random

filename = ''
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("error: unspecified input file", file = sys.stderr)
    sys.exit(-1)

num_lines = 0
if len(sys.argv) > 2:
    try:
        num_lines = int(sys.argv[2])
        if num_lines < 1:
            print("error: non-positive number of items", file = sys.stderr)
            sys.exit(-2)
    except ValueError:
        print("error: invalid number of items", file = sys.stderr)
        sys.exit(-2)
else:
    print("error: unspecified number of items", file = sys.stderr)
    sys.exit(-1)

try:
    with open(filename) as f:
        lines_in = [line.rstrip('\n') for line in f]
except FileNotFoundError:
    print("error: file '%s' not found" % filename, file = sys.stderr)
    sys.exit(-3)

if len(lines_in) == 0 or len(lines_in) == 1 and lines_in[0] == '':
    print("error: file '%s' is empty" % filename, file = sys.stderr)
    sys.exit(-3)

if num_lines > len(lines_in):
    num_lines = len(lines_in)
    print("warning: specified number of items exceeds number of lines in input file. Trimming number of items to %d." % num_lines)
lines_out = random.sample(lines_in, num_lines)
print(*lines_out)
sys.exit(0)
