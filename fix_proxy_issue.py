#!/usr/bin/env python3
"""
Fix for SGLang server proxy issue when using --host 0.0.0.0

Problem Analysis:
When using --host 0.0.0.0, the warmup request is being intercepted by a Squid proxy 
(172.16.37.200:3138) which blocks access to 0.0.0.0:25600 with a 403 Access Denied error.

Root Cause:
The warmup code in _execute_server_warmup() uses the host parameter directly to construct 
the URL (http://0.0.0.0:25600/get_model_info). When Python's requests library sees 0.0.0.0,
it may route through the system proxy, which blocks access to 0.0.0.0 addresses for security.
"""

import os
import sys
import subprocess
import json

def print_analysis():
    """Print detailed analysis of the issue"""
    print("=" * 80)
    print("SGLANG SERVER PROXY ISSUE ANALYSIS")
    print("=" * 80)
    print()
    print("PROBLEM:")
    print("  When starting SGLang server with --host 0.0.0.0, the warmup fails with:")
    print("  - 403 Access Denied from Squid proxy (172.16.37.200:3138)")
    print("  - Server starts successfully but warmup requests are blocked")
    print()
    print("ROOT CAUSE:")
    print("  1. SGLang's warmup code constructs URL using the --host parameter directly")
    print("  2. When --host is 0.0.0.0, warmup tries to connect to http://0.0.0.0:port")
    print("  3. Python requests library routes 0.0.0.0 through system proxy")
    print("  4. Corporate proxy (Squid) blocks access to 0.0.0.0 for security reasons")
    print()
    print("WHY IT WORKS WITH 127.0.0.1:")
    print("  - Localhost addresses (127.0.0.1) bypass proxy settings")
    print("  - Direct connection without proxy interference")
    print()

def provide_solutions():
    """Provide multiple solutions for the issue"""
    print("=" * 80)
    print("SOLUTIONS")
    print("=" * 80)
    print()
    
    print("SOLUTION 1: Disable proxy for SGLang process (RECOMMENDED)")
    print("-" * 40)
    print("Add these environment variables before starting SGLang:")
    print()
    print("export NO_PROXY=0.0.0.0,localhost,127.0.0.1")
    print("export no_proxy=0.0.0.0,localhost,127.0.0.1")
    print()
    print("Or in your startup script:")
    print()
    print("""NO_PROXY=0.0.0.0,localhost,127.0.0.1 \\
no_proxy=0.0.0.0,localhost,127.0.0.1 \\
$PYTHONHOME/bin/python3 -m sglang.launch_server \\
    --model-path $MODEL_PATH \\
    --host 0.0.0.0 \\
    --port 3000 \\
    --disable-cuda-graph \\
    --trust-remote-code \\
    --context-length 40960 \\
    --tp 8 \\
    --watchdog-timeout 600 \\
    --log-level debug \\
    2>&1 >> $SCRIPT_DIR/sglang.log &""")
    print()
    
    print("SOLUTION 2: Patch SGLang warmup code")
    print("-" * 40)
    print("Modify the warmup to use localhost for internal checks:")
    print("File: python/sglang/srt/entrypoints/http_server.py")
    print()
    print("Change line 1294 from:")
    print('    url = server_args.url()')
    print("To:")
    print('    # Use localhost for warmup even when binding to 0.0.0.0')
    print('    if server_args.host == "0.0.0.0":')
    print('        url = f"http://127.0.0.1:{server_args.port}"')
    print('    else:')
    print('        url = server_args.url()')
    print()
    
    print("SOLUTION 3: Skip server warmup")
    print("-" * 40)
    print("Add --skip-server-warmup flag (less recommended):")
    print()
    print("""$PYTHONHOME/bin/python3 -m sglang.launch_server \\
    --model-path $MODEL_PATH \\
    --host 0.0.0.0 \\
    --port 3000 \\
    --skip-server-warmup \\  # Add this flag
    --disable-cuda-graph \\
    --trust-remote-code \\
    --context-length 40960 \\
    --tp 8 \\
    --watchdog-timeout 600 \\
    --log-level debug \\
    2>&1 >> $SCRIPT_DIR/sglang.log &""")
    print()
    print("Note: Skipping warmup may affect initial performance")
    print()
    
    print("SOLUTION 4: Use specific IP address")
    print("-" * 40)
    print("Instead of 0.0.0.0, use your container's actual IP:")
    print()
    print("# Get container IP")
    print("hostname -I | awk '{print $1}'")
    print()
    print("# Use it in --host parameter")
    print("--host <your-container-ip>")
    print()

def create_patch_file():
    """Create a patch file for the warmup fix"""
    patch_content = """--- a/python/sglang/srt/entrypoints/http_server.py
+++ b/python/sglang/srt/entrypoints/http_server.py
@@ -1291,7 +1291,12 @@ def _execute_server_warmup(
     pipe_finish_writer: Optional[multiprocessing.connection.Connection],
 ):
     headers = {}
-    url = server_args.url()
+    # Use localhost for warmup when binding to 0.0.0.0 to avoid proxy issues
+    if server_args.host == "0.0.0.0":
+        url = f"http://127.0.0.1:{server_args.port}"
+    else:
+        url = server_args.url()
+    
     if server_args.api_key:
         headers["Authorization"] = f"Bearer {server_args.api_key}"
"""
    
    with open("/workspace/sglang_warmup_fix.patch", "w") as f:
        f.write(patch_content)
    
    print("PATCH FILE CREATED: /workspace/sglang_warmup_fix.patch")
    print("-" * 40)
    print("To apply the patch:")
    print("cd /workspace && patch -p1 < sglang_warmup_fix.patch")
    print()

def create_wrapper_script():
    """Create a wrapper script that handles the proxy issue automatically"""
    wrapper_content = """#!/bin/bash
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
"""
    
    with open("/workspace/sglang_server_wrapper.sh", "w") as f:
        f.write(wrapper_content)
    
    os.chmod("/workspace/sglang_server_wrapper.sh", 0o755)
    
    print("WRAPPER SCRIPT CREATED: /workspace/sglang_server_wrapper.sh")
    print("-" * 40)
    print("Usage:")
    print("""/workspace/sglang_server_wrapper.sh \\
    --model-path $MODEL_PATH \\
    --host 0.0.0.0 \\
    --port 3000 \\
    --disable-cuda-graph \\
    --trust-remote-code \\
    --context-length 40960 \\
    --tp 8 \\
    --watchdog-timeout 600 \\
    --log-level debug \\
    2>&1 >> $SCRIPT_DIR/sglang.log &""")
    print()

def main():
    print_analysis()
    provide_solutions()
    print("=" * 80)
    print("CREATING FIX FILES")
    print("=" * 80)
    print()
    create_patch_file()
    create_wrapper_script()
    
    print("=" * 80)
    print("IMMEDIATE WORKAROUND")
    print("=" * 80)
    print()
    print("For immediate use without modifying code, run:")
    print()
    print("NO_PROXY=0.0.0.0,localhost,127.0.0.1 \\")
    print("no_proxy=0.0.0.0,localhost,127.0.0.1 \\")
    print("  <your original sglang command>")
    print()
    print("This will bypass the proxy for local addresses.")
    print()

if __name__ == "__main__":
    main()