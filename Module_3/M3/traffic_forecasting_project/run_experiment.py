import argparse
from pathlib import Path
from directed_course.Module_3.M3.traffic_forecasting_project.src.pipeline import run

def main():
    parser = argparse.ArgumentParser(description="Traffic volume forecasting benchmark")
    parser.add_argument("--fast", action="store_true", help="Run a reduced smoke-test experiment")
    parser.add_argument("--origins", type=int, default=None, help="Maximum rolling test origins")
    args = parser.parse_args()
    run(fast=args.fast, max_origins=args.origins)

if __name__ == "__main__":
    main()
 