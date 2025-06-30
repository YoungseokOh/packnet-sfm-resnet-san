
import onnx
import sys

def analyze_onnx(model_path):
    """
    Analyzes an ONNX model, checking its validity and printing information
    about its inputs and outputs.

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file.
    """
    try:
        # Load the ONNX model
        model = onnx.load(model_path)

        # Check if the model is valid
        onnx.checker.check_model(model)

        print(f"Successfully loaded and checked ONNX model: {model_path}")
        print("-" * 30)

        # Print model inputs
        print("Model Inputs:")
        for input_tensor in model.graph.input:
            print(f"  - Name: {input_tensor.name}")
            # Extract shape and data type
            tensor_type = input_tensor.type.tensor_type
            shape = [dim.dim_value for dim in tensor_type.shape.dim]
            elem_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
            print(f"    - Data Type: {elem_type}")
            print(f"    - Shape: {shape}")
        print("-" * 30)

        # Print model outputs
        print("Model Outputs:")
        for output_tensor in model.graph.output:
            print(f"  - Name: {output_tensor.name}")
            # Extract shape and data type
            tensor_type = output_tensor.type.tensor_type
            shape = [dim.dim_value for dim in tensor_type.shape.dim]
            elem_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
            print(f"    - Data Type: {elem_type}")
            print(f"    - Shape: {shape}")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
    except onnx.checker.ValidationError as e:
        print(f"Error: ONNX model validation failed: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_onnx.py <path_to_onnx_model>", file=sys.stderr)
        sys.exit(1)
    
    model_path = sys.argv[1]
    analyze_onnx(model_path)
