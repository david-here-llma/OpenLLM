"""Allow 'python3 -m openllm', with 'selftest' handled straight away."""

import sys

from .cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "selftest":
        from .selftest import run
        sys.exit(run(argv[1:]))
    sys.exit(main())
