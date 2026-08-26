import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


DEFAULT_LAYER_OUTPUTS = {
    "layer2": "/layer2/layer2.3/relu_2/Relu_output_0",
    "layer3": "/layer3/layer3.5/relu_2/Relu_output_0",
}


def find_value_info(model: onnx.ModelProto, tensor_name: str):
    candidates = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for value in candidates:
        if value.name == tensor_name:
            return value
    return None


def make_output_value_info(model: onnx.ModelProto, output_name: str, source_tensor: str):
    value_info = find_value_info(model, source_tensor)
    if value_info is None:
        return helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)

    tensor_type = value_info.type.tensor_type
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return helper.make_tensor_value_info(output_name, tensor_type.elem_type, shape)


def main():
    parser = argparse.ArgumentParser(description="Export PatchCore layer2/layer3 outputs from a ResNet ONNX model.")
    parser.add_argument("--input", default="deployment/model/wide_resnet50_2.onnx")
    parser.add_argument("--output", default="deployment/model/wide_resnet50_2_patchcore.onnx")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    model = onnx.load(str(input_path))
    inferred = shape_inference.infer_shapes(model)

    existing_outputs = {output.name for output in inferred.graph.output}
    node_outputs = {name for node in inferred.graph.node for name in node.output}

    for output_name, source_tensor in DEFAULT_LAYER_OUTPUTS.items():
        if output_name in existing_outputs:
            continue
        if source_tensor not in node_outputs:
            raise RuntimeError(f"Cannot find source tensor for {output_name}: {source_tensor}")

        identity_node = helper.make_node(
            "Identity",
            inputs=[source_tensor],
            outputs=[output_name],
            name=f"PatchCore_{output_name}",
        )
        inferred.graph.node.append(identity_node)
        inferred.graph.output.append(make_output_value_info(inferred, output_name, source_tensor))
        existing_outputs.add(output_name)

    onnx.checker.check_model(inferred)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, str(output_path))

    print(f"Saved: {output_path}")
    print("Outputs:")
    for output in inferred.graph.output:
        dims = []
        for dim in output.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else (dim.dim_param or "?"))
        print(f"  {output.name}: {dims}")


if __name__ == "__main__":
    main()
