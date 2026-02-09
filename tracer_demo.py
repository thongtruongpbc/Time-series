from NicerTrace import NicerTrace, Tee
import sys, os, socket

cwd = os.path.realpath(".")
trace_output_file = f"{cwd}/trace.txt"

sys.stdout = Tee(trace_output_file)

tracer = NicerTrace(
    trace=1,
    count=1,
    timing=True,
    packages_to_include=["basicts"]
)

tracer.run('exec(open("demo.py").read())')
