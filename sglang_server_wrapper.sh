#!/bin/bash
# SGLang server wrapper script with proxy fix

# Disable proxy for local addresses
export NO_PROXY="0.0.0.0,localhost,127.0.0.1,*.local"
export no_proxy="0.0.0.0,localhost,127.0.0.1,*.local"

# Also unset proxy variables if needed (uncomment if required)
# unset HTTP_PROXY
# unset HTTPS_PROXY
# unset http_proxy
# unset https_proxy

echo "Starting SGLang server with proxy bypass..."
echo "NO_PROXY=$NO_PROXY"

# Pass all arguments to sglang
exec $PYTHONHOME/bin/python3 -m sglang.launch_server "$@"
