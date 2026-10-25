"""Energy CPU preparation functions used by the installed examples."""
from megakernel_lab.fast_examples.vendor.energy import cops4b
CPU_TASKS = {"capture": cops4b.cpu_capture, "rng": cops4b.cpu_rng, "recipe": cops4b.cpu_recipe, "data": cops4b.cpu_data}
