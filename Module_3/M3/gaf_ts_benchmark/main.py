import argparse
from directed_course.Module_3.M3.gaf_ts_benchmark.src.benchmark import prepare_all, run_benchmark

def main():
    p = argparse.ArgumentParser(description="GAF-TS multi-domain benchmark")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--domain", choices=["traffic", "energy", "weather", "all"], default="all")
    p.add_argument("--max-entities", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=None)
    args = p.parse_args()

    if not args.prepare and not args.benchmark:
        args.prepare = True
        args.benchmark = True

    if args.prepare:
        prepare_all()
    if args.benchmark:
        run_benchmark(
            domain=args.domain,
            max_entities=args.max_entities,
            max_windows=args.max_windows
        )

if __name__ == "__main__":
    main()
