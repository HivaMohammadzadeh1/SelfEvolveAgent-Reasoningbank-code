#!/usr/bin/env python3
"""
Comprehensive Log Analysis for ReasoningBank
Analyzes all log files to extract performance metrics, error patterns, and insights
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import statistics

# Paths
BASE_DIR = Path("/Users/hivamoh/Desktop/ReasoningBank")

class LogAnalyzer:
    def __init__(self):
        self.stats = {
            'webarena': defaultdict(lambda: defaultdict(list)),
            'swebench': defaultdict(lambda: defaultdict(list)),
            'mind2web': defaultdict(lambda: defaultdict(list)),
            'errors': Counter(),
            'warnings': Counter(),
            'timing': defaultdict(list),
            'tokens': defaultdict(list),
        }

    def parse_log_file(self, log_path):
        """Parse a single log file and extract metrics"""
        content = log_path.read_text(errors='ignore')

        # Extract basic info
        info = {
            'path': str(log_path),
            'name': log_path.name,
            'size_kb': log_path.stat().st_size / 1024,
            'lines': content.count('\n')
        }

        # Count log levels
        info['errors'] = content.count('| ERROR')
        info['warnings'] = content.count('| WARNING')
        info['infos'] = content.count('| INFO')

        # Extract errors
        error_pattern = r'\d{4}-\d{2}-\d{2}.*?\| ERROR\s+\|\s+(.+?)(?=\n\d{4}|\Z)'
        errors = re.findall(error_pattern, content, re.DOTALL)

        # Extract warnings
        warning_pattern = r'\d{4}-\d{2}-\d{2}.*?\| WARNING\s+\|\s+(.+?)(?=\n\d{4}|\Z)'
        warnings = re.findall(warning_pattern, content, re.DOTALL)

        # Extract timing info
        timing_pattern = r'walltime[:\s]+(\d+\.?\d*)'
        timings = [float(t) for t in re.findall(timing_pattern, content)]

        # Extract token usage
        token_pattern = r'tokens[_\s]+(?:input|output)[:\s]+(\d+)'
        tokens = [int(t) for t in re.findall(token_pattern, content)]

        # Extract success/failure
        success_pattern = r'success[:\s]+(true|false|True|False)'
        successes = re.findall(success_pattern, content, re.IGNORECASE)

        # Extract step counts
        step_pattern = r'steps[:\s]+(\d+)'
        steps = [int(s) for s in re.findall(step_pattern, content)]

        info['errors_list'] = errors[:10]  # First 10 errors
        info['warnings_list'] = warnings[:10]  # First 10 warnings
        info['timings'] = timings
        info['tokens'] = tokens
        info['successes'] = successes
        info['steps'] = steps

        return info

    def analyze_webarena_logs(self):
        """Analyze WebArena logs"""
        print("\n" + "="*70)
        print("ANALYZING WEBARENA LOGS")
        print("="*70)

        # Find WebArena log files
        webarena_logs = []
        for log_dir in [BASE_DIR / "logs-good", BASE_DIR / "logs_together", BASE_DIR / "logs_pro"]:
            if log_dir.exists():
                webarena_logs.extend(log_dir.glob("*multi*.log"))
                webarena_logs.extend(log_dir.glob("*reddit*.log"))

        results = defaultdict(lambda: {
            'total_files': 0,
            'total_size_mb': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'common_errors': Counter(),
            'common_warnings': Counter(),
            'avg_timing': [],
            'avg_tokens': [],
            'success_count': 0,
            'failure_count': 0,
        })

        for log_path in webarena_logs:
            print(f"\nAnalyzing: {log_path.name}")

            # Determine subset and mode
            if 'multi' in log_path.name:
                subset = 'multi'
            elif 'reddit' in log_path.name:
                subset = 'reddit'
            else:
                subset = 'other'

            if 'no_memory' in log_path.name:
                mode = 'no_memory'
            elif 'reasoningbank' in log_path.name:
                mode = 'reasoningbank'
            else:
                mode = 'unknown'

            key = f"{subset}_{mode}"

            try:
                info = self.parse_log_file(log_path)

                results[key]['total_files'] += 1
                results[key]['total_size_mb'] += info['size_kb'] / 1024
                results[key]['total_errors'] += info['errors']
                results[key]['total_warnings'] += info['warnings']

                # Count errors and warnings
                for error in info['errors_list']:
                    error_key = error[:100]  # First 100 chars
                    results[key]['common_errors'][error_key] += 1

                for warning in info['warnings_list']:
                    warning_key = warning[:100]
                    results[key]['common_warnings'][warning_key] += 1

                # Aggregate metrics
                results[key]['avg_timing'].extend(info['timings'])
                results[key]['avg_tokens'].extend(info['tokens'])

                # Count successes
                for success in info['successes']:
                    if success.lower() == 'true':
                        results[key]['success_count'] += 1
                    else:
                        results[key]['failure_count'] += 1

                print(f"  Lines: {info['lines']}, Errors: {info['errors']}, Warnings: {info['warnings']}")
                if info['timings']:
                    print(f"  Avg timing: {statistics.mean(info['timings']):.2f}s")
                if info['tokens']:
                    print(f"  Total tokens: {sum(info['tokens'])}")

            except Exception as e:
                print(f"  ERROR parsing {log_path.name}: {e}")

        return results

    def analyze_swebench_logs(self):
        """Analyze SWE-Bench logs"""
        print("\n" + "="*70)
        print("ANALYZING SWE-BENCH LOGS")
        print("="*70)

        swebench_dirs = [
            BASE_DIR / "logs-good/swebench_no_memory",
            BASE_DIR / "logs-good/swebench_reasoningbank",
        ]

        results = defaultdict(lambda: {
            'total_tasks': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'repos': Counter(),
            'error_types': Counter(),
            'avg_timing': [],
        })

        for log_dir in swebench_dirs:
            if not log_dir.exists():
                continue

            mode = 'no_memory' if 'no_memory' in log_dir.name else 'reasoningbank'
            print(f"\nAnalyzing {mode} logs from {log_dir.name}")

            task_logs = list(log_dir.glob("task_*.log"))
            print(f"Found {len(task_logs)} task logs")

            for log_path in task_logs[:50]:  # Analyze first 50 for speed
                try:
                    info = self.parse_log_file(log_path)

                    results[mode]['total_tasks'] += 1
                    results[mode]['total_errors'] += info['errors']
                    results[mode]['total_warnings'] += info['warnings']

                    # Extract repo name
                    repo_match = re.match(r'task_([^_]+)__', log_path.name)
                    if repo_match:
                        repo = repo_match.group(1)
                        results[mode]['repos'][repo] += 1

                    # Aggregate timings
                    results[mode]['avg_timing'].extend(info['timings'])

                    # Count error types
                    for error in info['errors_list']:
                        if 'timeout' in error.lower():
                            results[mode]['error_types']['timeout'] += 1
                        elif 'connection' in error.lower():
                            results[mode]['error_types']['connection'] += 1
                        elif 'parse' in error.lower() or 'json' in error.lower():
                            results[mode]['error_types']['parse'] += 1
                        else:
                            results[mode]['error_types']['other'] += 1

                except Exception as e:
                    print(f"  ERROR parsing {log_path.name}: {e}")

        return results

    def generate_report(self, webarena_results, swebench_results):
        """Generate comprehensive analysis report"""
        report = []
        report.append("="*70)
        report.append("COMPREHENSIVE LOG ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*70)

        # WebArena Summary
        report.append("\n" + "="*70)
        report.append("WEBARENA ANALYSIS SUMMARY")
        report.append("="*70)

        for key, data in sorted(webarena_results.items()):
            report.append(f"\n{key.upper().replace('_', ' ')}:")
            report.append(f"  Files analyzed: {data['total_files']}")
            report.append(f"  Total size: {data['total_size_mb']:.2f} MB")
            report.append(f"  Total errors: {data['total_errors']}")
            report.append(f"  Total warnings: {data['total_warnings']}")

            if data['success_count'] + data['failure_count'] > 0:
                total = data['success_count'] + data['failure_count']
                sr = data['success_count'] / total * 100
                report.append(f"  Success rate: {sr:.2f}% ({data['success_count']}/{total})")

            if data['avg_timing']:
                report.append(f"  Avg timing: {statistics.mean(data['avg_timing']):.2f}s")
                report.append(f"  Median timing: {statistics.median(data['avg_timing']):.2f}s")

            if data['avg_tokens']:
                report.append(f"  Total tokens: {sum(data['avg_tokens']):,}")
                report.append(f"  Avg tokens: {statistics.mean(data['avg_tokens']):.0f}")

            if data['common_errors']:
                report.append("\n  Top 3 errors:")
                for error, count in data['common_errors'].most_common(3):
                    report.append(f"    [{count}x] {error[:80]}...")

            if data['common_warnings']:
                report.append("\n  Top 3 warnings:")
                for warning, count in data['common_warnings'].most_common(3):
                    report.append(f"    [{count}x] {warning[:80]}...")

        # SWE-Bench Summary
        report.append("\n" + "="*70)
        report.append("SWE-BENCH ANALYSIS SUMMARY")
        report.append("="*70)

        for mode, data in sorted(swebench_results.items()):
            report.append(f"\n{mode.upper()}:")
            report.append(f"  Tasks analyzed: {data['total_tasks']}")
            report.append(f"  Total errors: {data['total_errors']}")
            report.append(f"  Total warnings: {data['total_warnings']}")

            if data['avg_timing']:
                report.append(f"  Avg timing: {statistics.mean(data['avg_timing']):.2f}s")

            if data['repos']:
                report.append("\n  Top repositories:")
                for repo, count in data['repos'].most_common(5):
                    report.append(f"    {repo}: {count} tasks")

            if data['error_types']:
                report.append("\n  Error types:")
                for error_type, count in data['error_types'].most_common():
                    report.append(f"    {error_type}: {count}")

        # Key Insights
        report.append("\n" + "="*70)
        report.append("KEY INSIGHTS")
        report.append("="*70)

        # Compare no_memory vs reasoningbank
        multi_nomem = webarena_results.get('multi_no_memory', {})
        multi_rbank = webarena_results.get('multi_reasoningbank', {})

        if multi_nomem.get('success_count') is not None and multi_rbank.get('success_count') is not None:
            nomem_total = multi_nomem['success_count'] + multi_nomem['failure_count']
            rbank_total = multi_rbank['success_count'] + multi_rbank['failure_count']

            if nomem_total > 0 and rbank_total > 0:
                nomem_sr = multi_nomem['success_count'] / nomem_total * 100
                rbank_sr = multi_rbank['success_count'] / rbank_total * 100
                improvement = rbank_sr - nomem_sr

                report.append(f"\n1. Multi subset: {nomem_sr:.1f}% (no mem) → {rbank_sr:.1f}% (RBank)")
                report.append(f"   Improvement: +{improvement:.1f} percentage points")

        # Error analysis
        total_errors = sum(d['total_errors'] for d in webarena_results.values())
        total_warnings = sum(d['total_warnings'] for d in webarena_results.values())

        report.append(f"\n2. Total errors across all WebArena logs: {total_errors}")
        report.append(f"   Total warnings: {total_warnings}")

        # Timing comparison
        nomem_times = multi_nomem.get('avg_timing', [])
        rbank_times = multi_rbank.get('avg_timing', [])

        if nomem_times and rbank_times:
            speedup = statistics.mean(nomem_times) / statistics.mean(rbank_times)
            report.append(f"\n3. Average speedup with memory: {speedup:.2f}x")

        return "\n".join(report)

def main():
    """Run comprehensive log analysis"""
    print("="*70)
    print("REASONINGBANK COMPREHENSIVE LOG ANALYSIS")
    print("="*70)

    analyzer = LogAnalyzer()

    # Analyze WebArena logs
    webarena_results = analyzer.analyze_webarena_logs()

    # Analyze SWE-Bench logs
    swebench_results = analyzer.analyze_swebench_logs()

    # Generate report
    report = analyzer.generate_report(webarena_results, swebench_results)

    # Print report
    print("\n" * 2)
    print(report)

    # Save report
    output_path = BASE_DIR / "LOG_ANALYSIS_REPORT.txt"
    output_path.write_text(report)
    print(f"\n✅ Report saved to: {output_path}")

    # Save detailed JSON
    json_output = {
        'webarena': {k: {
            'total_files': v['total_files'],
            'total_size_mb': v['total_size_mb'],
            'total_errors': v['total_errors'],
            'total_warnings': v['total_warnings'],
            'success_rate': v['success_count'] / (v['success_count'] + v['failure_count']) * 100
                if v['success_count'] + v['failure_count'] > 0 else 0,
            'avg_timing': statistics.mean(v['avg_timing']) if v['avg_timing'] else 0,
            'total_tokens': sum(v['avg_tokens']) if v['avg_tokens'] else 0,
        } for k, v in webarena_results.items()},
        'swebench': {k: {
            'total_tasks': v['total_tasks'],
            'total_errors': v['total_errors'],
            'total_warnings': v['total_warnings'],
            'repos': dict(v['repos']),
            'error_types': dict(v['error_types']),
            'avg_timing': statistics.mean(v['avg_timing']) if v['avg_timing'] else 0,
        } for k, v in swebench_results.items()}
    }

    json_path = BASE_DIR / "LOG_ANALYSIS_DATA.json"
    json_path.write_text(json.dumps(json_output, indent=2))
    print(f"✅ Detailed data saved to: {json_path}")

if __name__ == "__main__":
    main()
