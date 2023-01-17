from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("myimage/1.png")
# print(img)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("totensor", img_tensor, 1)

# 归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("nomalize", img_norm, 2)

# resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("resize", img_resize, 3)

# compose resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize2", img_resize_2, 4)

# Randomcrop
trans_random = transforms.RandomCrop((500, 100))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("randomCrop", img_crop, i)



writer.close()
