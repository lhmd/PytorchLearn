from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# tensor数据类型
# transforms.ToTensor解决两个问题
# 1. transform如何使用
# 2. tensor数据类型和普通有什么区别

image_path = "test_dataset/train/ants_image/0013035.jpg"
img = Image.open(image_path)

writer = SummaryWriter("logs")

# 创建具体对象
tensor_trans = transforms.ToTensor()
# 使用具体对象
tensor_image = tensor_trans(img)

writer.add_image("Tensor_img", tensor_image)
writer.close()
