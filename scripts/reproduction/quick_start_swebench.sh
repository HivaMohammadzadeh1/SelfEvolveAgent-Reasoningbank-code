#!/bin/bash
# Quick start script for SWE-bench evaluation with ReasoningBank
# This script reproduces Table 2 from the paper

set -e  # Exit on error

echo "=================================================="
echo "ReasoningBank - SWE-bench Quick Start"
echo "Reproducing Table 2 from the paper"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
    echo "Creating template .env file..."
    cat > .env << EOF
# Add your API keys here
GOOGLE_API_KEY=your_google_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
EOF
    echo "✓ Created .env template"
    echo "⚠️  Please edit .env and add your API keys before continuing"
    exit 1
fi

# Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.swebench.txt
echo "✓ Dependencies installed"
echo ""

# Test mode or full evaluation
if [ "$1" == "--test" ]; then
    echo "Running in TEST MODE (limited to 10 tasks)"
    MAX_TASKS="--max_tasks 10"
    echo ""
else
    echo "Running FULL EVALUATION (500 tasks - this will take several hours)"
    echo "To run a quick test instead, use: ./quick_start_swebench.sh --test"
    MAX_TASKS=""
    echo ""

    # Ask for confirmation
    read -p "Continue with full evaluation? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Run with --test flag for a quick test."
        exit 1
    fi
fi

# Choose model
echo "Select model:"
echo "  1) gemini-2.5-flash (recommended, faster)"
echo "  2) gemini-2.5-pro (higher quality)"
read -p "Enter choice (1-2): " MODEL_CHOICE
echo ""

case $MODEL_CHOICE in
    1)
        MODEL="gemini-2.5-flash"
        ;;
    2)
        MODEL="gemini-2.5-pro"
        ;;
    *)
        echo "Invalid choice. Using gemini-2.5-flash"
        MODEL="gemini-2.5-flash"
        ;;
esac

echo "Using model: $MODEL"
echo ""

# Run experiments
echo "=================================================="
echo "Running Experiments"
echo "=================================================="
echo ""

# Update config with selected model
python -c "
import yaml
with open('config.swebench.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['llm']['model'] = '$MODEL'
with open('config.swebench.yaml', 'w') as f:
    yaml.dump(config, f)
"

echo "Experiment 1/3: No-Memory Baseline"
python run_swebench.py --mode no_memory --config config.swebench.yaml $MAX_TASKS
echo "✓ No-Memory complete"
echo ""

echo "Experiment 2/3: Synapse Baseline"
python run_swebench.py --mode synapse --config config.swebench.yaml $MAX_TASKS
echo "✓ Synapse complete"
echo ""

echo "Experiment 3/3: ReasoningBank"
python run_swebench.py --mode reasoningbank --config config.swebench.yaml $MAX_TASKS
echo "✓ ReasoningBank complete"
echo ""

# Aggregate results
echo "=================================================="
echo "Aggregating Results"
echo "=================================================="
echo ""

python reproduce_table2.py --aggregate_only --model $MODEL

echo ""
echo "=================================================="
echo "Evaluation Complete!"
echo "=================================================="
echo ""
echo "Results saved to: results/swebench/"
echo "Comparison table: results/swebench/table2_comparison.csv"
echo ""
echo "To view logs:"
echo "  ls logs/swebench/"
echo ""
echo "Expected results from paper (Table 2):"
if [ "$MODEL" == "gemini-2.5-flash" ]; then
    echo "  No Memory:      34.2% resolve rate, 30.3 steps"
    echo "  Synapse:        35.4% resolve rate, 30.7 steps"
    echo "  ReasoningBank:  38.8% resolve rate, 27.5 steps"
else
    echo "  No Memory:      54.0% resolve rate, 21.1 steps"
    echo "  Synapse:        53.4% resolve rate, 21.0 steps"
    echo "  ReasoningBank:  57.4% resolve rate, 19.8 steps"
fi
echo ""
echo "For full documentation, see README_SWEBENCH.md"
