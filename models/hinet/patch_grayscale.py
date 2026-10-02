from pathlib import Path
import shutil

file = Path("models/hinet/HINet/basicsr/models/archs/hinet_arch.py")
source = file.read_text(encoding="utf-8")

old = """class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
"""
new = """class SAM(nn.Module):
    def __init__(self, n_feat, in_chn=3, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, in_chn, kernel_size, bias=bias)
        self.conv3 = conv(in_chn, n_feat, kernel_size, bias=bias)
"""

if old in source:
    shutil.copy2(file, file.with_suffix(".py.backup"))
    source = source.replace(old, new)
    source = source.replace("self.sam12 = SAM(prev_channels)", "self.sam12 = SAM(prev_channels, in_chn=in_chn)")
    file.write_text(source, encoding="utf-8")
    print("HINet grayscale change applied. Original file backed up.")
else:
    print("Original SAM code not found. Check hinet_arch.py before changing anything.")
