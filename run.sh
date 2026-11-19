#!/bin/bash
# Wrapper script to execute app.py using available python3 runtime

BUNDLED_PY="/Users/tomasinahill/Library/Caches/antigravity/python/6de4e3f5ffe707ea/bin/python3"

if [ -f "$BUNDLED_PY" ]; then
    PY_BIN="$BUNDLED_PY"
elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
else
    echo "Error: python3 not found."
    exit 1
fi

if [ "$1" = "test" ] || [ "$1" = "--test" ]; then
    "$PY_BIN" -m unittest discover -s tests -p "test_*.py" -v
elif [ "$1" = "streamlit" ] || [ "$1" = "--streamlit" ]; then
    shift
    if command -v streamlit >/dev/null 2>&1; then
        streamlit run streamlit_app.py "$@"
    else
        "$PY_BIN" -m streamlit run streamlit_app.py "$@"
    fi
else
    "$PY_BIN" app.py "$@"
fi
