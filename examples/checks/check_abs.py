"""Trusted C89 demo validator. Not a full ACOM lab acceptance suite."""

import re
import subprocess
import sys
from pathlib import Path


def main():
    source = Path("solution.c")
    if not source.exists():
        print("Missing solution.c")
        return 1
    text = source.read_text(encoding="utf-8")
    # A simple screening rule, not a C parser or proof of libc independence.
    cleaned = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    if re.search(r"#\s*include|\b(?:fabs|abs|malloc|calloc|realloc|free|memcpy|memset)\s*\(", cleaned):
        print("Forbidden include or library call in solution.c")
        return 1
    build = Path(".arpo-check")
    build.mkdir(exist_ok=True)
    harness = build / "harness.c"
    harness.write_text(
        """#include <stdio.h>
extern double lab_abs(double x);
int main(void) {
    double xs[] = {0.0, -0.0, 1.0, -1.0, 1.25, -1.25, 1e-200, -1e-200, 1e200, -1e200};
    unsigned int i;
    double expected, actual;
    for (i = 0; i < sizeof(xs) / sizeof(xs[0]); ++i) {
        expected = xs[i] == 0.0 ? 0.0 : (xs[i] < 0.0 ? -xs[i] : xs[i]);
        actual = lab_abs(xs[i]);
        if (actual != expected || (actual == 0.0 && 1.0 / actual < 0.0)) {
            printf("FAIL input %.17g expected %.17g (both zero signs must give +0), output %.17g\\n",
                xs[i], expected, actual);
            return 1;
        }
    }
    printf("PASS 10 numerical cases including signed zero\\n");
    return 0;
}
""",
        encoding="utf-8",
    )
    # The Apple /usr/bin shim tries to cache xcrun data outside the workspace.
    # Prefer the actual compiler when Command Line Tools are installed.
    apple_clang = Path("/Library/Developer/CommandLineTools/usr/bin/clang")
    compiler = str(apple_clang) if sys.platform == "darwin" and apple_clang.is_file() else "clang"
    argv = [
        compiler,
        "-std=c89",
        "-pedantic-errors",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source),
        str(harness),
        "-o",
        str(build / "test"),
    ]
    apple_sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    if compiler == str(apple_clang) and apple_sdk.is_dir():
        argv[1:1] = ["-isysroot", str(apple_sdk)]
    compile_result = subprocess.run(argv, check=False)
    if compile_result.returncode:
        return compile_result.returncode
    return subprocess.run([str((build / "test").resolve())], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
