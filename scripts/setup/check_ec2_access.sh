#!/bin/bash
# Quick check for EC2 access requirements

HOSTNAME="ec2-3-150-230-27.us-east-2.compute.amazonaws.com"

echo "================================"
echo "EC2 Access Checklist"
echo "================================"
echo ""

# Check 1: SSH keys
echo "1. Checking for SSH keys..."
if ls ~/.ssh/*.pem >/dev/null 2>&1; then
    echo "   ✅ Found SSH key(s):"
    ls -1 ~/.ssh/*.pem 2>/dev/null | head -5
    echo ""
    echo "   To connect, use:"
    echo "   ssh -i ~/.ssh/YOUR-KEY.pem ubuntu@$HOSTNAME"
else
    echo "   ❌ No .pem files found in ~/.ssh/"
    echo "   You need to get the SSH key for this EC2 instance"
fi
echo ""

# # Check 2: Can reach host (basic ping/connection)
# echo "2. Checking if EC2 instance is reachable..."
# if nc -z -w5 ${HOSTNAME#http://} 22 2>/dev/null; then
#     echo "   ✅ Port 22 (SSH) is open"
# elif timeout 5 bash -c "cat < /dev/null > /dev/tcp/${HOSTNAME#http://}/22" 2>/dev/null; then
#     echo "   ✅ Port 22 (SSH) is open"
# else
#     echo "   ❌ Cannot reach port 22 (SSH)"
#     echo "   Possible issues:"
#     echo "   - EC2 instance is not running"
#     echo "   - Security group doesn't allow your IP"
#     echo "   - Network connectivity issue"
# fi
# echo ""

# Check 3: Environment setup
echo "3. Checking local environment..."
if [ -f "venv/bin/activate" ]; then
    echo "   ✅ Virtual environment exists"
else
    echo "   ❌ Virtual environment not found"
fi

if [ -f "setup_ec2_env.sh" ]; then
    echo "   ✅ EC2 environment script ready"
else
    echo "   ❌ EC2 environment script not found"
fi
echo ""

# Check 4: BrowserGym installation
echo "4. Checking BrowserGym installation..."
if source venv/bin/activate 2>/dev/null && python -c "import browsergym" 2>/dev/null; then
    echo "   ✅ BrowserGym is installed"
else
    echo "   ⚠️  BrowserGym may need installation"
    echo "   Run: ./setup_browsergym.sh"
fi
echo ""

echo "================================"
echo "Next Steps"
echo "================================"
echo ""
echo "1. Make sure you have the SSH key (.pem file)"
echo "2. Read the full guide: EC2_SETUP_INSTRUCTIONS.md"
echo "3. Connect to EC2 and start services"
echo "4. Test connection: ./test_ec2_connection.sh"
echo "5. Run your first test"
echo ""

