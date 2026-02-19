import matplotlib.pyplot as plt
import csv
import os

def plot():
    # Load Data
    csv_path = os.path.join(os.path.dirname(__file__), "../benchmark_results/results.csv")
    kernels = []
    speedups = []
    
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                if "Slower" not in row[3]:
                    kernels.append(row[0])
                    # Extract number from "287x"
                    val = float(row[3].replace("x", "").replace(",", ""))
                    speedups.append(val)
    except FileNotFoundError:
        print("Error: Run main.py first to generate results.csv!")
        return

    # Create Chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(kernels, speedups, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    # Styling
    plt.yscale('log') # Log scale because MatMul (1666x) dwarfs everything else
    plt.title("The 'Interpreter Tax': Python vs C++ (Log Scale)", fontsize=16, fontweight='bold')
    plt.ylabel("Speedup Factor (x Times Faster)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x',
                ha='center', va='bottom', fontweight='bold')

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "../benchmark_results/performance_chart.png")
    plt.savefig(out_path)
    print(f"Graph saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    plot()
