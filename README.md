从.npz文件直接转为tensor：
1.将 PIL.Image 或 numpy.ndarray 数据转换为 torch.FloatTensor。
如果输入是一个 PIL.Image 对象，则使用 torch.from_numpy(np.asarray(pic)) 将图像数据转换为 numpy.ndarray，并进一步使用 .float() 将其转换为 torch.FloatTensor；如果输入是一个 numpy.ndarray 对象，则直接将其转换为 torch.FloatTensor。
2.将像素值从 [0, 255] 的 uint8 类型转换为 [0.0, 1.0] 的 float 类型。这个操作通过将每个像素值除以 255 来完成。"img = img.div(255.0)"
3.交换维度顺序，即从 (H, W, C) 变为 (C, H, W)。这个操作使用 tensor.permute() 方法来实现
"img.permute((2, 0, 1))"



PF model测试图片到噪声的模型的效果
Glow：注释代码
Glow_test:修改代码框架，让输出的噪声和图片都是相同维度的
Model:用Glow_test的框架训练从图片到噪声的模型,但是修改的很浅显，只修改了n_Block=1
Model2:好像没啥用，就是用来测试64*64的正常情况
Model3:修改了最后输出的噪声的形状,用reshape的方式将5个噪声修改成1个噪声
Model4:用Model3的结果进行训练从图片到噪声的模型,且--affine设为false
Model4_t:用于输出
Model5:每一个block后都reshape,且--affine设为false
Model5_t:用于输出
Model6和model6t:用于测试
model6u:输入改为噪声、输出改为图片，同时对输出的图片进行reshape，也就是没改内部的结构
Model7:输入改为噪声、输出改为图片，修改内部的结构。
Modle8:输入和输出都不变,修改内部的结构。
One:用Unet model训练从图片到噪声的模型


CUDA_VISIBLE_DEVICES=3 python train.py  "../dataset/train"
batch_size越大，显存占用空间越多，但是训练效果越好。

我们直接修改Glow的损失函数：输出的数据是图片，输出的数据是噪声，
然后直接用Glow的框架，训练从图片到噪声的函数。这时候loss就不需要追求最小似然比，
而是用Glow生成的噪声和我们的噪声的差距。

为什么可以这么做：Glow模型的目的是为了让图片对应到一个高斯分布上，对应过去的所有数据点的概率之积最大，这样达到图片对应的数据符合高斯分布。而我们已经构造好从图片到高斯分布的一个目标，所以我们直接让对应过去即可。

Glow_test用于写代码
CUDA_VISIBLE_DEVICES=7 python train.py  "../dataset/train" --affine

输出的噪声：
list中有4个，形状分别为:8为batch_size
torch.Size([8, 6, 16, 16])
torch.Size([8, 12, 8, 8])
torch.Size([8, 24, 4, 4])
torch.Size([8, 96, 2, 2])
但是我们想要的是torch.Size([8, 3, 32, 32])，

原因出在了：
"RealNVP在NICE的基础上的另一大改进就是做多尺度框架的设计。所谓的多尺度就是映射得到的最终潜在空间不是一次得到的，而是通过不同尺度的潜在变量拼接而成的。"Glow就是基于RealNVP框架进行修改的，因此我们要把这一项修改回去。

接下来画出模型的数据通路，然后进行改造。同时测试改造完的模型的效果。


尝试一个一个修改模型的参数，达到让噪声规模不变，但是改了Affine模块和AffineCoupling模块后，下一个flow模型的Affine模型又对不上了。

受不了了，直接resize:
"""
zero = torch.zeros_like(out)
mean, log_sd = self.prior(zero).chunk(2, 1)
log_p = gaussian_log_p(out, mean, log_sd)
log_p = log_p.view(b_size, -1).sum(1)
out=out.reshape(b_size, n_channel, height, width)#这里重塑z_new和out的形状
z_new = out
"""
同时把所有block都设置为split=Flase的类型，然后连在一起

原来的图片是torch.Size([4, 3, 256, 256])、且每一个数都是[0,1]，所以需要放大乘以255
而直接用.npz文件，是已经放大255倍了，所以不需要再乘以255

mpiexec -n 1 python image_sample.py --batch_size 4 --generator determ-indiv --training_mode consistency_distillation --sampler onestep --model_path ../../model_path/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 100 --resblock_updown True --use_fp16 True --weight_schedule uniform


glow训练的是K=1,L=40的--affine
model2:K=1,L=120,affine=true 不行
model5:重新训练，K=4，L=20，没有affine



model6:
去掉了            
if args.n_bits < 8:
image = torch.floor(image / 2 ** (8 - args.n_bits))
看是否会影响训练效果

model6_t:验证我们的图片处理方式是否正确