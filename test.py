# System libs
import os
import argparse
# Numerical libs
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from scipy.io import loadmat
from scipy.misc import imread, imresize, imsave, imshow
from scipy.ndimage import zoom
# Our libs
from models import ModelBuilder
from utils import colorEncode
import time

### define colors
colors = np.array([[0, 0, 0], [0, 0, 0], [255, 255, 255]],dtype=np.uint8)


# forward func for testing
def forward_test_multiscale(nets, img, args):
    (net_encoder, net_decoder) = nets

    pred = torch.zeros(1, args.num_class, img.size(2), img.size(3))
    pred = Variable(pred, volatile=True).cuda()

    for scale in args.scales:
        img_scale = zoom(img.numpy(),
                         (1., 1., scale, scale),
                         order=1,
                         prefilter=False,
                         mode='nearest')

        # feed input data
        input_img = Variable(torch.from_numpy(img_scale),
                             volatile=True).cuda()

        # forward
        # x= net_encoder(input_img)
        # pred = net_decoder(x)

        # forward
        pred_scale = net_decoder(net_encoder(input_img),
                                 segSize=(img.size(2), img.size(3)))


        # average the probability
        pred = pred + pred_scale / len(args.scales)

    return pred



def visualize_test_result(img,seg, pred, args):

    img = img[0]
    lab=imread(gt_img,mode='RGB')
    pred = pred.data.cpu()[0]
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    img = imresize(img, (args.imgSize, args.imgSize),
                    interp='bilinear')

    # segmentation
    # lab = (lab.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    # lab_color = colorEncode(lab, colors[0])
    lab_color = imresize(lab, (args.imgSize, args.imgSize),
                            interp='nearest')

    # prediction
    pred_ = np.argmax(pred.numpy(), axis=0) + 1
    pred_color = colorEncode(pred_, colors)
    pred_color = imresize(pred_color, (args.imgSize, args.imgSize),
                            interp='nearest')
    
    # aggregate images and save
    im_vis = np.concatenate((img, lab_color, pred_color), axis=1).astype(np.uint8)
    
    # imshow(im_vis)
    
    imsave(os.path.join(args.vis,os.path.basename(test_img)), im_vis)

    imsave(os.path.join(args.result,os.path.basename(test_img)), pred_)


def test(nets, args):
    # switch to eval mode
    for net in nets:
        net.eval()

    # loading image, resize, convert to tensor
    img = imread(test_img, mode='RGB')
    h, w = img.shape[0], img.shape[1]
    s = 1. * args.imgSize / min(h, w)
    img = imresize(img, s)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img = img_transform(img)
    img = img.view(1, img.size(0), img.size(1), img.size(2))

    seg = imread(gt_img,mode='RGB')
 
    # forward pass
    pred = forward_test_multiscale(nets, img, args)

    # visualization
    visualize_test_result(img, seg, pred, args)


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.arch_encoder,
                                        fc_dim=args.fc_dim,
                                        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(arch=args.arch_decoder,
                                        fc_dim=args.fc_dim,
                                        segSize=args.segSize,
                                        weights=args.weights_decoder,
                                        use_softmax=True)
    
    nets = (net_encoder, net_decoder)
    # print(net)
    for net in nets:
        net.cuda()

    tic=time.time()
    # single pass
    test(nets, args)
    # print(nets)
    toc=time.time()
    print('time:', toc-tic)
    print('Done! Output is saved in {}'.format(args.result))


if __name__ == '__main__':
    root_dir='/media/mostafa/CODE/mostafa_Lesion_Segmentation_final/MICCAI2018/2016/data/384x384/Test/OR/'
    seg_dir='/media/mostafa/CODE/mostafa_Lesion_Segmentation_final/MICCAI2018/2016/data/384x384/Test/GT/'
    for img in os.listdir(root_dir):
        print (img)
        main_part=img.split('.')[0]
        test_img=root_dir+img
        gt_img=seg_dir+main_part+'.png'
        print(gt_img)
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--id', default='384x384',
                            help="a name for identifying the model to load")
        parser.add_argument('--suffix', default='_best.pth',
                            help="which snapshot to load")
        parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                            help="architecture of net_encoder")
        parser.add_argument('--arch_decoder', default='psp_bilinear',
                            help="architecture of net_decoder")
        parser.add_argument('--fc_dim', default=2048, type=int,
                            help='number of features between encoder and decoder')

        # Data related arguments
        parser.add_argument('--num_val', default=-1, type=int,
                            help='number of images to evalutate')
        parser.add_argument('--num_class', default=2, type=int,
                            help='number of classes')
        parser.add_argument('--batch_size', default=20, type=int,
                            help='batchsize')
        parser.add_argument('--imgSize', default=384, type=int,
                            help='resize input image')
        parser.add_argument('--segSize', default=384, type=int,
                            help='output image size, -1 = keep original')

        # Misc arguments  
        parser.add_argument('--ckpt', default='./ckpt/MY_2017_noEPE',
                            help='folder to output checkpoints')
        parser.add_argument('--vis', default='./2016/SLSDeep-EPE/vis',
                        help='folder to output visualization during training')
        parser.add_argument('--result', default='./2016/SLSDeep-EPE/results',
                            help='folder to output visualization results')

        args = parser.parse_args()
        print(args)

        args.vis = os.path.join(args.vis, args.id)

        # scales for evaluation
        args.scales = (0.5, 0.75, 1, 1.25, 1.5)

        # absolute paths of model weights
        args.weights_encoder = os.path.join(args.ckpt, args.id,
                                            'encoder' + args.suffix)
        args.weights_decoder = os.path.join(args.ckpt, args.id,
                                            'decoder' + args.suffix)

        main(args)