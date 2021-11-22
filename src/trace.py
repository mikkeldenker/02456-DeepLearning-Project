import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import sys

class QuantModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self._base_model = base_model
        self._quant = torch.quantization.QuantStub()
        self._dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self._quant(x)
        x = self._base_model(x)
        x = self._dequant(x)

        return x

torch.backends.quantized.engine = 'qnnpack'

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=3,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
example_input = torch.randn(3, 1280, 760)
model.load_state_dict(torch.load("../model.pth"))
# model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear})
# model.backbone = QuantModel(model.backbone)
# model = torch.quantization.fuse_modules(model, [['conv', 'batchnorm', 'relu']])
# model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
# model = torch.quantization.prepare(model)
# model = torch.quantization.convert(model)
model.eval()

print(model(example_input.unsqueeze(0)))

torch.save(model.state_dict(), "../quant_model.pth")

sys.exit(0)

traced_model = torch.jit.script(model)



print(traced_model)
torch.jit.save(traced_model, '../traced.pth')

