import cProfile, pstats
import pstats, math
import io


profiler = cProfile.Profile()
profiler.enable()
import camoc_agent.trainer

profiler.disable()

result = io.StringIO()
pstats.Stats(profiler, stream=result).print_stats()
result = result.getvalue()
result = "ncalls" + result.split("ncalls")[-1]
result = "\n".join(
    [",".join(line.rstrip().split(None, 5)) for line in result.split("\n")]
)

with open("test.csv", "w+") as f:
    f.write(result)
    f.close()
