import pytest, tempfile, os
import train

def test_training():
    with tempfile.TemporaryDirectory() as tmp_dir:
        train.train(output_path=os.path.join(tmp_dir, 'test.onnx'), data_points_per_epoch=4, epochs=1)