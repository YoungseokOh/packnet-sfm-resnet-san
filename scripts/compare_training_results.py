#!/usr/bin/env python3
"""
Automatic Training Results Comparison
Automatically finds and compares evaluation results from checkpoint directories
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def find_evaluation_results(checkpoint_base_dir):
    """
    Recursively find evaluation_results folders in checkpoint directory
    Returns dict: {experiment_name: evaluation_results_path}
    """
    checkpoint_base = Path(checkpoint_base_dir)
    results_map = {}
    
    # Look for evaluation_results folders
    for eval_dir in checkpoint_base.rglob("evaluation_results"):
        if eval_dir.is_dir():
            # Get experiment name (parent directory name)
            experiment_name = eval_dir.parent.name
            results_map[experiment_name] = eval_dir
            
    return results_map

def load_all_epochs(eval_results_dir):
    """
    Load all epoch results from evaluation_results directory
    Returns dict: {epoch_num: {metric: value}}
    """
    epochs_data = {}
    eval_path = Path(eval_results_dir)
    
    for result_file in sorted(eval_path.glob("epoch_*_results.json")):
        # Extract epoch number from filename
        epoch_num = int(result_file.stem.split('_')[1])
        
        with open(result_file, 'r') as f:
            data = json.load(f)
            epochs_data[epoch_num] = {
                "abs_rel": data.get("ncdb-cls-640x384-combined_val-abs_rel", None),
                "rmse": data.get("ncdb-cls-640x384-combined_val-rmse", None),
                "a1": data.get("ncdb-cls-640x384-combined_val-a1", None),
                "sqr_rel": data.get("ncdb-cls-640x384-combined_val-sqr_rel", None),
                "rmse_log": data.get("ncdb-cls-640x384-combined_val-rmse_log", None),
                "a2": data.get("ncdb-cls-640x384-combined_val-a2", None),
                "a3": data.get("ncdb-cls-640x384-combined_val-a3", None),
            }
    
    return epochs_data

def print_experiment_summary(experiment_name, epochs_data):
    """Print summary statistics for an experiment"""
    if not epochs_data:
        print(f"  No data available")
        return
    
    epochs = sorted(epochs_data.keys())
    abs_rel_values = [epochs_data[e]["abs_rel"] for e in epochs if epochs_data[e]["abs_rel"] is not None]
    
    if abs_rel_values:
        first_epoch = epochs[0]
        last_epoch = epochs[-1]
        best_epoch = min(epochs, key=lambda e: epochs_data[e]["abs_rel"])
        
        first_abs_rel = epochs_data[first_epoch]["abs_rel"]
        last_abs_rel = epochs_data[last_epoch]["abs_rel"]
        best_abs_rel = epochs_data[best_epoch]["abs_rel"]
        
        improvement = ((first_abs_rel - last_abs_rel) / first_abs_rel) * 100
        
        print(f"  Epochs: {len(epochs)} ({epochs[0]} to {epochs[-1]})")
        print(f"  First epoch {first_epoch}: abs_rel={first_abs_rel:.6f}")
        print(f"  Last epoch {last_epoch}:  abs_rel={last_abs_rel:.6f}")
        print(f"  Best epoch {best_epoch}:  abs_rel={best_abs_rel:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")

def compare_experiments(experiments_data):
    """
    Compare multiple experiments
    experiments_data: dict of {experiment_name: epochs_data}
    """
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)
    
    # Find common epochs
    all_epochs = [set(exp_data.keys()) for exp_data in experiments_data.values()]
    common_epochs = sorted(set.intersection(*all_epochs)) if all_epochs else []
    
    if not common_epochs:
        print("\n⚠️  No common epochs found across experiments")
        return
    
    print(f"\nCommon epochs: {len(common_epochs)} epochs ({min(common_epochs)} to {max(common_epochs)})")
    
    # Print comparison table
    exp_names = list(experiments_data.keys())
    print("\n" + "=" * 120)
    print(f"{'Epoch':<8}", end="")
    for name in exp_names:
        print(f"{name[:35]:<40}", end="")
    print()
    print(f"{'':8}", end="")
    for _ in exp_names:
        print(f"{'abs_rel':<12} {'rmse':<12} {'a1 %':<12}", end="")
    print()
    print("=" * 120)
    
    for epoch in common_epochs:
        print(f"{epoch:<8}", end="")
        for name in exp_names:
            data = experiments_data[name][epoch]
            abs_rel = data["abs_rel"] if data["abs_rel"] is not None else float('nan')
            rmse = data["rmse"] if data["rmse"] is not None else float('nan')
            a1 = data["a1"] * 100 if data["a1"] is not None else float('nan')
            print(f"{abs_rel:<12.6f} {rmse:<12.4f} {a1:<12.2f}", end="")
        print()
    
    # Calculate average metrics
    print("\n" + "=" * 100)
    print("AVERAGE METRICS (Common Epochs)")
    print("=" * 100)
    print(f"{'Experiment':<40} {'avg abs_rel':<15} {'avg rmse':<15} {'avg a1':<15}")
    print("-" * 100)
    
    for name in exp_names:
        abs_rel_values = [experiments_data[name][e]["abs_rel"] 
                         for e in common_epochs 
                         if experiments_data[name][e]["abs_rel"] is not None]
        rmse_values = [experiments_data[name][e]["rmse"] 
                      for e in common_epochs 
                      if experiments_data[name][e]["rmse"] is not None]
        a1_values = [experiments_data[name][e]["a1"] 
                    for e in common_epochs 
                    if experiments_data[name][e]["a1"] is not None]
        
        avg_abs_rel = np.mean(abs_rel_values) if abs_rel_values else float('nan')
        avg_rmse = np.mean(rmse_values) if rmse_values else float('nan')
        avg_a1 = np.mean(a1_values) if a1_values else float('nan')
        
        print(f"{name:<40} {avg_abs_rel:<15.6f} {avg_rmse:<15.6f} {avg_a1:<15.6f}")
    
    # Best performance
    print("\n" + "=" * 100)
    print("BEST PERFORMANCE")
    print("=" * 100)
    
    for name in exp_names:
        epochs = sorted(experiments_data[name].keys())
        if epochs:
            best_epoch = min(epochs, key=lambda e: experiments_data[name][e]["abs_rel"])
            best_data = experiments_data[name][best_epoch]
            print(f"\n{name}:")
            print(f"  Best Epoch: {best_epoch}")
            print(f"  abs_rel: {best_data['abs_rel']:.6f}")
            print(f"  rmse:    {best_data['rmse']:.6f}")
            print(f"  a1:      {best_data['a1']:.6f}")

def plot_comparison(experiments_data, output_prefix="training_comparison"):
    """
    Create comprehensive comparison plots
    """
    # Create outputs/compare_training_results directory if it doesn't exist
    output_dir = Path("outputs/compare_training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Experiments Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFB380', '#B8A2C4', '#FFE66D']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    
    exp_names = list(experiments_data.keys())
    
    # Plot 1: abs_rel
    ax1 = axes[0, 0]
    for i, (name, data) in enumerate(experiments_data.items()):
        epochs = sorted(data.keys())
        abs_rel = [data[e]["abs_rel"] for e in epochs if data[e]["abs_rel"] is not None]
        ax1.plot(epochs, abs_rel, marker=markers[i % len(markers)], 
                label=name[:30], linewidth=2.5, markersize=6, 
                color=colors[i % len(colors)], alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('abs_rel (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Relative Error', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: RMSE
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(experiments_data.items()):
        epochs = sorted(data.keys())
        rmse = [data[e]["rmse"] for e in epochs if data[e]["rmse"] is not None]
        ax2.plot(epochs, rmse, marker=markers[i % len(markers)], 
                label=name[:30], linewidth=2.5, markersize=6, 
                color=colors[i % len(colors)], alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: a1 accuracy
    ax3 = axes[1, 0]
    for i, (name, data) in enumerate(experiments_data.items()):
        epochs = sorted(data.keys())
        a1 = [data[e]["a1"] * 100 for e in epochs if data[e]["a1"] is not None]
        ax3.plot(epochs, a1, marker=markers[i % len(markers)], 
                label=name[:30], linewidth=2.5, markersize=6, 
                color=colors[i % len(colors)], alpha=0.8)
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('a1 Accuracy % (Higher is Better)', fontsize=12, fontweight='bold')
    ax3.set_title('δ < 1.25 Accuracy', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Performance gap (if 2 experiments)
    ax4 = axes[1, 1]
    if len(exp_names) == 2:
        name1, name2 = exp_names
        data1, data2 = experiments_data[name1], experiments_data[name2]
        common_epochs = sorted(set(data1.keys()) & set(data2.keys()))
        
        if common_epochs:
            gaps = []
            for e in common_epochs:
                if data1[e]["abs_rel"] is not None and data2[e]["abs_rel"] is not None:
                    gap = ((data1[e]["abs_rel"] - data2[e]["abs_rel"]) / data1[e]["abs_rel"]) * 100
                    gaps.append(gap)
                else:
                    gaps.append(None)
            
            # Filter out None values
            valid_epochs = [e for e, g in zip(common_epochs, gaps) if g is not None]
            valid_gaps = [g for g in gaps if g is not None]
            
            if valid_gaps:
                colors_bars = ['#95E1D3' if g > 0 else '#FFB380' for g in valid_gaps]
                ax4.bar(valid_epochs, valid_gaps, color=colors_bars, alpha=0.7, 
                       edgecolor='black', linewidth=1.5)
                ax4.axhline(y=0, color='black', linestyle='-', linewidth=2)
                
                avg_gap = np.mean(valid_gaps)
                ax4.axhline(y=avg_gap, color='red', linestyle='--', linewidth=2.5, 
                           label=f'Average: {avg_gap:+.2f}%', alpha=0.8)
                
                ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                ax4.set_ylabel(f'Gap % ({name2[:15]} vs {name1[:15]})', fontsize=11, fontweight='bold')
                ax4.set_title(f'Performance Gap\n(Positive = {name2[:20]} Better)', 
                            fontsize=12, fontweight='bold')
                ax4.legend(fontsize=9)
                ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    else:
        # Show training progression comparison
        for i, (name, data) in enumerate(experiments_data.items()):
            epochs = sorted(data.keys())
            if len(epochs) >= 2:
                abs_rel_values = [data[e]["abs_rel"] for e in epochs if data[e]["abs_rel"] is not None]
                if len(abs_rel_values) >= 2:
                    improvements = []
                    for j in range(1, len(abs_rel_values)):
                        imp = ((abs_rel_values[j-1] - abs_rel_values[j]) / abs_rel_values[j-1]) * 100
                        improvements.append(imp)
                    
                    ax4.plot(epochs[1:], improvements, marker=markers[i % len(markers)], 
                            label=name[:30], linewidth=2, markersize=5, 
                            color=colors[i % len(colors)], alpha=0.8)
        
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Epoch-to-Epoch Improvement %', fontsize=11, fontweight='bold')
        ax4.set_title('Training Progression', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save to outputs directory
    output_path_png = output_dir / f'{output_prefix}.png'
    output_path_pdf = output_dir / f'{output_prefix}.pdf'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"\n✅ Plots saved to {output_path_png} and {output_path_pdf}")

def main():
    print("=" * 100)
    print("AUTOMATIC TRAINING RESULTS COMPARISON")
    print("=" * 100)
    
    # Define checkpoint directories to search
    checkpoint_dirs = [
        "/workspace/packnet-sfm/checkpoints/resnetsan01_adaptive_multi_domain_v2",
        "/workspace/packnet-sfm/checkpoints/resnetsan01_640x384_newest_0.05to100"
    ]
    
    # Find all evaluation_results directories
    all_experiments = {}
    
    print("\n🔍 Searching for evaluation results...")
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"  ⚠️  Directory not found: {checkpoint_dir}")
            continue
        
        # Get the parent directory name as experiment label
        parent_name = checkpoint_path.name
        print(f"\n📁 {parent_name}/")
        
        results_map = find_evaluation_results(checkpoint_dir)
        
        if not results_map:
            print(f"  ⚠️  No evaluation_results found")
            continue
        
        for exp_name, eval_path in results_map.items():
            # Create a more descriptive name
            full_name = f"{parent_name}"
            print(f"  ✓ Found: {eval_path}")
            
            # Load all epochs
            epochs_data = load_all_epochs(eval_path)
            if epochs_data:
                all_experiments[full_name] = epochs_data
                print(f"    Loaded {len(epochs_data)} epochs")
    
    if not all_experiments:
        print("\n❌ No experiments found!")
        return
    
    # Print summary for each experiment
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARIES")
    print("=" * 100)
    
    for name, data in all_experiments.items():
        print(f"\n🔬 {name}:")
        print_experiment_summary(name, data)
    
    # Compare experiments
    if len(all_experiments) >= 2:
        compare_experiments(all_experiments)
    
    # Create plots
    print("\n" + "=" * 100)
    print("GENERATING PLOTS")
    print("=" * 100)
    plot_comparison(all_experiments, "automatic_training_comparison")
    
    print("\n" + "=" * 100)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 100)

if __name__ == "__main__":
    main()
