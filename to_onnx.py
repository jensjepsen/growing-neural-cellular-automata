import torch, onnxruntime as ort, onnx, numpy as np, sys
import model

checkpoint_path, output_path = sys.argv[1:3]

m = model.Model(); m.load_state_dict(torch.load(checkpoint_path)['state_dict'])

m.export_onnx(output_path, verbose=True)

onnx_model = onnx.load(output_path)

session = ort.InferenceSession(output_path)

output = session.run(None, {
    'steps': np.array([100]),
    'batch_size': np.array([1]),
    'initial_state': np.zeros((1, 40, 40, 16), dtype=np.float32)
})
