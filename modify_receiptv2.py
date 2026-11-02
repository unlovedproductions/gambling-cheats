#!/usr/bin/env python3
# Same six-block preview as modify_receipt.py. v2 was truncated and exited
# without prompting; this file is the runnable copy.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modify_receipt import main

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)
