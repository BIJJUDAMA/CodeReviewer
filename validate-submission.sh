#!/bin/bash

# Configuration
PING_URL=${HF_SPACE_URL:-"http://localhost:7860"}

echo "--- Starting Validation ---"

# 1. Docker Build Check
echo "[1/3] Checking Docker build..."
docker build -t test-env server/
if [ $? -eq 0 ]; then
    echo "SUCCESS: Docker build passed."
else
    echo "FAILURE: Docker build failed."
    exit 1
fi

# 2. API Reset Check
echo "[2/3] Checking /reset endpoint..."
# Start the server in background for testing if not running
# (In a real CI this would be pre-deployed)
RESPONSE=$(curl -s -X POST $PING_URL/reset -d '{}')
if [[ $RESPONSE == *"code_snippet"* ]]; then
    echo "SUCCESS: /reset returned valid observation."
else
    echo "FAILURE: /reset failed or returned invalid JSON. Response: $RESPONSE"
    echo "Hint: Make sure the server is running at $PING_URL"
    exit 1
fi

# 3. OpenEnv Specification Check
echo "[3/3] Validating openenv.yaml..."
if [ -f "openenv.yaml" ]; then
    echo "SUCCESS: openenv.yaml exists."
    # Basic grep validation for mandatory fields
    grep -q "name:" openenv.yaml && grep -q "tasks:" openenv.yaml && grep -q "endpoints:" openenv.yaml
    if [ $? -eq 0 ]; then
        echo "SUCCESS: openenv.yaml contains required fields."
    else
        echo "FAILURE: openenv.yaml is missing required fields."
        exit 1
    fi
else
    echo "FAILURE: openenv.yaml not found."
    exit 1
fi

echo "--- All Checks Passed! ---"
