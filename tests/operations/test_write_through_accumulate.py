"""A factored matrix store must preserve the region output's accumulation contract."""
from __future__ import annotations

from megakernel_lab.operation_compiler.lower import _lower_serial


def _transposed_result():
    # This is the weight VJP form produced by F.linear: mm -> t -> dense clone.
    # It is a region test, independent of loop capture and model names.
    shapes = {"a": (64, 16), "b": (16, 64), "product": (64, 64),
              "view": (64, 64), "result": (64, 64)}
    values = {name: {"shape": list(shape), "strides": [shape[1], 1], "dtype": "float32"}
              for name, shape in shapes.items()}
    values["view"]["strides"] = [1, 64]
    return {"values": values, "inputs": ["a", "b"], "outputs": ["result"], "operations": [
        {"name": "product", "target": "aten.mm.default", "args": [{"ref": "a"}, {"ref": "b"}], "kwargs": {}},
        {"name": "view", "target": "aten.t.default", "args": [{"ref": "product"}], "kwargs": {}},
        {"name": "result", "target": "aten.clone.default", "args": [{"ref": "view"}],
         "kwargs": {"memory_format": "contiguous_format"}}]}


def _lower(binding):
    return _lower_serial(_transposed_result(), output_bindings=[binding], region_body=True,
        barrier_schedule="dependency", staged=True, write_through=True,
        matrix_engine={"accumulate": True})


def test_write_through_keeps_accumulating_destination():
    program = _lower("accumulate")
    manifest = program.manifest
    assert manifest["output_placement"]["result"]["mode"] == "accumulate"
    decisions = manifest["matrix_decisions"]
    assert any(d["output"] == "product" and d["selected"] for d in decisions)
    assert not any(d.get("write_through") == "result" for d in decisions)
    assert "v_result[i] + r_result" in program.kernel_body
    assert any(0 in stage["accumulates"] for stage in manifest["stages"])


def test_write_through_still_places_store_destination():
    program = _lower("store")
    assert any(d.get("write_through") == "result" for d in program.manifest["matrix_decisions"])
    assert program.manifest["output_placement"]["result"]["mode"] == "store"
    assert "v_result[i] + r_result" not in program.kernel_body


if __name__ == "__main__":
    test_write_through_keeps_accumulating_destination()
    test_write_through_still_places_store_destination()
    print("PASS: accumulated and stored factored matrix outputs")
