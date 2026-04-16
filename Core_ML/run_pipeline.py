import subprocess
import os

scripts = [
    "ML/preprocessing.py",
    "ML/drift_detection.py",
    "ML/behavior_analysis.py",
    "ML/model_evaluation.py"
]

def run_all():
    print("=== STARTING ML & DRIFT ANALYSIS PIPELINE ===")
    for script in scripts:
        print(f"\n>> Executing {script}...")
        result = subprocess.run(["python", script], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"[!] Error in {script}:")
            print(result.stderr)
            break
    print("=== PIPELINE EXECUTION COMPLETE ===")

if __name__ == "__main__":
    run_all()
