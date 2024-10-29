import torch
import torch.nn as nn
import torch.onnx

# y = 1.0 - x を計算する単純なネットワークを定義
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return 1.0 - x

# モデルをインスタンス化
model = Model()

# ダミー入力を定義（ONNXにエクスポートするために必要）
dummy_input = torch.rand(3, 640, 640)

# モデルをONNX形式で保存
torch.onnx.export(
    model,               # エクスポートするモデル
    dummy_input,         # ダミー入力
    "model_3x640x640.onnx",   # 保存するONNXファイル名
    export_params=True,  # パラメータも一緒にエクスポート
    opset_version=12,    # ONNXのバージョン
    do_constant_folding=True,  # 定数フォールディングを有効化
    input_names=['input'],   # 入力名
    output_names=['output']  # 出力名
)

print("モデルがONNX形式で保存されました。")

