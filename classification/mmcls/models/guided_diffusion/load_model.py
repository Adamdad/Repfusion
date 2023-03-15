import torch
from mmcls.models.guided_diffusion.script_util import (
    args_to_dict, create_model_and_diffusion, model_and_diffusion_defaults)
# from torchvision.models.feature_extraction import create_feature_extractor
from mmcls.models.classifiers.kd_ddpm import ForwardHookManager


def create_imagenet256(model_path=""):
    defaults = model_and_diffusion_defaults()
    imagenet256_dict = dict(
        attention_resolutions = '32,16,8',
        class_cond = False ,
        diffusion_steps =1000 ,
        image_size =256 ,
        learn_sigma =True ,
        noise_schedule ='linear' ,
        num_channels =256 ,
        num_head_channels =64 ,
        num_res_blocks =2 ,
        resblock_updown =True ,
        use_fp16 =False ,
        use_scale_shift_norm =True,
    )
    defaults.update(imagenet256_dict)
    print("Create Model imagenet256")
    model, diffusion = create_model_and_diffusion(**defaults)
    
    print(f"Load Checkpoint {model_path}")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    
    return model

def create_imagenet64(model_path=""):
    defaults = model_and_diffusion_defaults()
    imagenet64_dict = dict(
        attention_resolutions = '32,16,8',
        class_cond = True,
        diffusion_steps =1000 ,
        image_size =64,
        learn_sigma =True,
        noise_schedule ='cosine',
        num_channels =192,
        num_head_channels =64 ,
        num_res_blocks =3 ,
        resblock_updown =True ,
        use_fp16 =False ,
        use_scale_shift_norm =True,
    )
    # --attention_resolutions 32,16,8 
    # --class_cond True 
    # --diffusion_steps 1000 
    # --dropout 0.1 
    # --image_size 64 
    # --learn_sigma True 
    # --noise_schedule cosine 
    # --num_channels 192 
    # --num_head_channels 64 
    # --num_res_blocks 3 
    # --resblock_updown True 
    # --use_new_attention_order True 
    # --use_fp16 True 
    # --use_scale_shift_norm True
    defaults.update(imagenet64_dict)
    print("Create Model imagenet64")
    model, diffusion = create_model_and_diffusion(**defaults)
    
    print(f"Load Checkpoint {model_path}")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    
    return model

if __name__ == '__main__':
    model = create_imagenet64('/home/yangxingyi/guided-diffusion/64x64_diffusion.pt')
    for name, module in model.named_modules():
        print(name)

    layer_name = 'middle_block.2.out_layers.3'


    # Now you can build the feature extractor. This returns a module whose forward
    # method returns a dictionary like:
    # {
    #     'layer1': output of layer 1,
    #     'layer2': output of layer 2,
    #     'layer3': output of layer 3,
    #     'layer4': output of layer 4,
    # }
    hoods = ForwardHookManager()
    hoods.add_hook(model, layer_name)
    timesteps = torch.tensor([100])
    y = torch.tensor([100])
    # # model = create_feature_extractor(model, return_nodes=return_nodes)
    out = model(torch.rand(1, 3, 64, 64), timesteps, y)
    # print(out)
    print([hook.feat.shape for hook in hoods.hook_list])
