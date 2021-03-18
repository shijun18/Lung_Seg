import segmentation_models_pytorch as smp


if __name__ == "__main__":

   
    net = smp.Unet(
        encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                     # model output channels (number of classes in your dataset)
        aux_params={"classes":1} 
    )

    from torchsummary import summary
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    summary(net.cuda(),input_size=(1,512,512),batch_size=1,device='cuda')
    # summary(net.cuda(),input_size=(1,256,256),batch_size=1,device='cuda')
    # import sys
    # sys.path.append('..')
    # from utils import count_params_and_macs
    # count_params_and_macs(net.cuda(),(1,1,256,256))
    