#!/bin/bash
# WebArena Environment Configuration for EC2 Instance
# Updated on 2025-10-27
# 
# Usage: source setup_ec2_env.sh

export SHOPPING="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7770"
export SHOPPING_ADMIN="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7780"
export REDDIT="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:9999"
export GITLAB="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8023"
export MAP="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:3000"
export WIKIPEDIA="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8888"
export HOMEPAGE="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:4399"

# For BrowserGym/WebArena compatibility
export WA_SHOPPING="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7770"
export WA_SHOPPING_ADMIN="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7780/admin"
export WA_REDDIT="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:9999"
export WA_GITLAB="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8023"
export WA_MAP="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:3000"
export WA_WIKIPEDIA="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_HOMEPAGE="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:4399"

echo "✅ WebArena environment variables set for EC2 instance"
echo "   Hostname: ec2-3-150-230-27.us-east-2.compute.amazonaws.com"

