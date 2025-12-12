#!/usr/bin/env python3
"""
Test script to verify WebArena data loading.

Usage:
    python scripts/test_webarena_loader.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.webarena_loader import WebArenaDataset
from loguru import logger


def test_loading():
    """Test loading WebArena data."""
    
    print("="*60)
    print("WebArena Data Loader Test")
    print("="*60)
    print()
    
    # Test loading
    print("1. Testing data loader...")
    try:
        loader = WebArenaDataset(data_dir="data/webarena")
        print("   ✓ Loader initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return False
    
    # Load admin tasks
    print("\n2. Loading Admin subset tasks...")
    try:
        tasks = loader.load_tasks(subsets=["admin"])
        print(f"   ✓ Loaded {len(tasks)} admin tasks")
        
        if len(tasks) == 0:
            print("   ✗ No tasks loaded - check if data exists")
            return False
            
    except FileNotFoundError as e:
        print(f"   ✗ Data not found: {e}")
        print("\n   Please run: bash scripts/download_webarena_data.sh")
        return False
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return False
    
    # Show statistics
    print("\n3. Dataset statistics:")
    stats = loader.get_statistics()
    print(f"   Total tasks: {stats['total_tasks']}")
    print(f"   Subsets: {stats['subsets']}")
    for subset, count in stats['subset_counts'].items():
        print(f"     - {subset}: {count} tasks")
    
    # Show sample tasks
    print("\n4. Sample admin tasks:")
    admin_tasks = loader.get_tasks_by_subset("admin")
    for i, task in enumerate(admin_tasks[:3], 1):
        print(f"\n   Task {i}:")
        print(f"     ID: {task['task_id']}")
        print(f"     Description: {task['description'][:80]}...")
        print(f"     Start URL: {task['start_url']}")
        print(f"     Requires login: {task['require_login']}")
    
    # Test all subsets
    print("\n5. Testing all subsets (except map)...")
    all_subsets = ["shopping", "admin", "gitlab", "reddit", "multi"]
    try:
        # Use max_multi_tasks=29 to match paper (WebArena has 48 total)
        all_tasks = loader.load_tasks(subsets=all_subsets, max_multi_tasks=29)
        print(f"   ✓ Loaded {len(all_tasks)} total tasks")
        
        # Expected counts from paper
        expected = {
            "shopping": 187,
            "admin": 182,
            "gitlab": 180,
            "reddit": 106,
            "multi": 29
        }
        
        print("\n   Comparison with paper:")
        for subset in all_subsets:
            actual = len([t for t in all_tasks if t["subset"] == subset])
            exp = expected[subset]
            match = "✓" if actual == exp else "⚠"
            print(f"     {match} {subset}: {actual} tasks (expected: {exp})")
            
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("Test completed successfully! ✓")
    print("="*60)
    print("\nYou can now run:")
    print("  python run_eval.py --mode no_memory --subset admin")
    print()
    
    return True


if __name__ == "__main__":
    success = test_loading()
    sys.exit(0 if success else 1)
