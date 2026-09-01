"""One-shot OpenSSH Askpass bridge for LabelForge's JUSUF MFA preflight."""

import os
import sys


if __name__ == "__main__":
    # The value is inherited for one child process only. Do not write files,
    # logs, settings or diagnostics from this deliberately tiny helper.
    sys.stdout.write(os.environ.get("LABELFORGE_TOTP", "") + "\n")
    sys.stdout.flush()
