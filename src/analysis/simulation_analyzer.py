"""
Analysis tools for government simulation results

Provides visualization and statistical analysis capabilities for simulation outputs.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import json


class SimulationAnalyzer:
    """Analyze simulation results and generate insights"""
    
    def __init__(self, history: List[Dict[str, Any]]):
        """
        Initialize analyzer with simulation history
        
        Args:
            history: List of state dictionaries from simulation
        """
        self.history = history
        self.metrics = self._extract_metrics()
    
    def _extract_metrics(self) -> Dict[str, List[float]]:
        """Extract time series metrics from history"""
        metrics = {
            'steps': [],
            'avg_satisfaction': [],
            'avg_wealth': [],
            'wealth_inequality': [],
            'budget': []
        }
        
        for state in self.history:
            for key in metrics.keys():
                if key in state:
                    metrics[key].append(state[key])
        
        return metrics
    
    def calculate_trends(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate trend statistics for key metrics
        
        Returns:
            Dictionary of metrics with their trend statistics
        """
        trends = {}
        
        for metric_name, values in self.metrics.items():
            if metric_name == 'steps' or len(values) < 2:
                continue
            
            # Calculate trend line using least squares
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            
            trends[metric_name] = {
                'slope': coeffs[0],
                'initial': values[0],
                'final': values[-1],
                'change': values[-1] - values[0],
                'percent_change': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return trends
    
    def identify_critical_points(self, metric: str = 'avg_satisfaction') -> List[Dict[str, Any]]:
        """
        Identify critical points where metric changes significantly
        
        Args:
            metric: The metric to analyze
            
        Returns:
            List of critical points with their characteristics
        """
        if metric not in self.metrics:
            return []
        
        values = self.metrics[metric]
        critical_points = []
        
        # Calculate rate of change
        if len(values) < 3:
            return critical_points
        
        changes = np.diff(values)
        threshold = np.std(changes) * 1.5
        
        for i, change in enumerate(changes):
            if abs(change) > threshold:
                critical_points.append({
                    'step': i + 1,
                    'value': values[i + 1],
                    'change': change,
                    'type': 'spike' if change > 0 else 'drop'
                })
        
        return critical_points
    
    def compare_scenarios(self, other_analyzer: 'SimulationAnalyzer') -> Dict[str, Any]:
        """
        Compare this simulation with another scenario
        
        Args:
            other_analyzer: Another SimulationAnalyzer to compare with
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        for metric in ['avg_satisfaction', 'avg_wealth', 'wealth_inequality']:
            if metric in self.metrics and metric in other_analyzer.metrics:
                self_final = self.metrics[metric][-1]
                other_final = other_analyzer.metrics[metric][-1]
                
                comparison[metric] = {
                    'scenario_a_final': self_final,
                    'scenario_b_final': other_final,
                    'difference': self_final - other_final,
                    'percent_difference': ((self_final - other_final) / other_final * 100) if other_final != 0 else 0
                }
        
        return comparison
    
    def generate_report(self) -> str:
        """
        Generate a text report of simulation analysis
        
        Returns:
            Formatted report string
        """
        report = ["=" * 60]
        report.append("SIMULATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 60)
        trends = self.calculate_trends()
        
        for metric, stats in trends.items():
            report.append(f"\n{metric.upper().replace('_', ' ')}:")
            report.append(f"  Initial: {stats['initial']:.2f}")
            report.append(f"  Final: {stats['final']:.2f}")
            report.append(f"  Change: {stats['change']:.2f} ({stats['percent_change']:.1f}%)")
            report.append(f"  Mean: {stats['mean']:.2f}")
            report.append(f"  Std Dev: {stats['std']:.2f}")
            report.append(f"  Trend: {stats['slope']:.4f} per step")
        
        # Critical points
        report.append("\n" + "-" * 60)
        report.append("CRITICAL POINTS")
        report.append("-" * 60)
        
        for metric in ['avg_satisfaction', 'wealth_inequality']:
            if metric in self.metrics:
                points = self.identify_critical_points(metric)
                if points:
                    report.append(f"\n{metric.upper().replace('_', ' ')}:")
                    for point in points:
                        report.append(f"  Step {point['step']}: {point['type'].upper()} "
                                    f"(change: {point['change']:.2f})")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def export_data(self, filepath: str) -> None:
        """
        Export analysis data to JSON file
        
        Args:
            filepath: Path to save JSON file
        """
        export_data = {
            'metrics': self.metrics,
            'trends': self.calculate_trends(),
            'critical_points': {
                metric: self.identify_critical_points(metric)
                for metric in ['avg_satisfaction', 'avg_wealth', 'wealth_inequality']
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


def analyze_policy_effectiveness(simulation_history: List[Dict[str, Any]], 
                                policy_implementation_steps: List[int]) -> Dict[str, Any]:
    """
    Analyze the effectiveness of policies by examining pre/post implementation metrics
    
    Args:
        simulation_history: Complete simulation history
        policy_implementation_steps: Steps at which policies were implemented
        
    Returns:
        Analysis of policy effectiveness
    """
    effectiveness = []
    
    for step in policy_implementation_steps:
        if step < 5 or step >= len(simulation_history) - 5:
            continue
        
        # Get before and after windows
        before = simulation_history[step - 5:step]
        after = simulation_history[step:step + 5]
        
        avg_satisfaction_before = np.mean([s['avg_satisfaction'] for s in before])
        avg_satisfaction_after = np.mean([s['avg_satisfaction'] for s in after])
        
        effectiveness.append({
            'step': step,
            'satisfaction_before': avg_satisfaction_before,
            'satisfaction_after': avg_satisfaction_after,
            'impact': avg_satisfaction_after - avg_satisfaction_before
        })
    
    return {
        'policy_impacts': effectiveness,
        'total_policies': len(policy_implementation_steps),
        'positive_impacts': sum(1 for e in effectiveness if e['impact'] > 0),
        'negative_impacts': sum(1 for e in effectiveness if e['impact'] < 0)
    }


if __name__ == "__main__":
    # Example usage with mock data
    mock_history = [
        {'step': i, 'avg_satisfaction': 50 + i * 0.5 + np.random.randn() * 2,
         'avg_wealth': 10000 + i * 100, 'wealth_inequality': 5000 - i * 10,
         'budget': 100000 - i * 1000}
        for i in range(50)
    ]
    
    analyzer = SimulationAnalyzer(mock_history)
    print(analyzer.generate_report())
