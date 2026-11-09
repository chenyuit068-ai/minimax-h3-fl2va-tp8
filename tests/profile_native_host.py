"""Diagnose host dispatch overhead; this is not a GPU performance score."""
import cProfile
import pstats
from pathlib import Path
import sys
import torch
import infinicore as core
from infinilm.lib import _infinilm as native
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_nine_runtime import NativeFusions

x=torch.ones((1,1024,5376),device='cuda',dtype=torch.bfloat16)
values=(x,x.clone(),x.clone());torch.cuda.synchronize()
args=tuple(core.from_torch(t)._underlying for t in values)
with NativeFusions(attention=False):
    def execute():return native._h3_test_dispatch('adaln',*args)
    execute();core.sync_device()
    profiler=cProfile.Profile();profiler.enable()
    for _ in range(300):execute()
    core.sync_device();profiler.disable()
    pstats.Stats(profiler).sort_stats('cumulative').print_stats(30)
