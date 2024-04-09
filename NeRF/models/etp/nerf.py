import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn
import math
import scipy.signal
import heapq
import cv2
from tqdm import tqdm

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


# Model
class NeRF(nn.Module):
    def __init__(self, D=4, W=256, input_ch=3, output_ch=4):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
      
        self.tcnn = tcnn.Network(
            n_input_dims=input_ch,
            n_output_dims=output_ch,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": W,
                "n_hidden_layers": D,
            },
        )

    def forward(self, x):
        outputs = self.tcnn(x)
        return outputs    




def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()


    # training options
    parser.add_argument("--near", type=float, default=0., 
                        help='near distance')
    parser.add_argument("--far", type=float, default=10., 
                        help='far distance')
    parser.add_argument("--camera_angle", type=float, default=90., 
                        help='camera FOV angle')
    parser.add_argument("--pointcloud_search_radius", type=float, default=0.2, 
                        help='pointcloud_search_radius')
    parser.add_argument("--featurecloud_search_radius", type=float, default=1., 
                        help='featurecloud_search_radius')
    parser.add_argument("--pointcloud_search_num", type=int, default=16, 
                        help='pointcloud_search_num')
    parser.add_argument("--featurecloud_search_num", type=int, default=4, 
                        help='featurecloud_search_num')
    parser.add_argument("--featuremap_scale", type=int, default=8, 
                        help='featuremap_scale')
    parser.add_argument("--chunk", type=int, default=1024, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--feature_loss_weight", type=float, default=0.01, 
                        help='weight of the language embedded feature loss')


    parser.add_argument("--rgba_net_layers", type=int, default=8, 
                        help='layers in rgb network')
    parser.add_argument("--rgba_net_width", type=int, default=512, 
                        help='channels per layer in rgb net')
    parser.add_argument("--clip_net_layers", type=int, default=8, 
                        help='layers in clip network')
    parser.add_argument("--clip_net_width", type=int, default=512, 
                        help='channels per layer in clip net')

    parser.add_argument("--N_rand", type=int, default=16*16, 
                        help='batch size (number of random rays per gradient step)')
   

    # rendering options
    parser.add_argument("--N_samples", type=int, default=256, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=8,
                        help='number of fine samples per ray')


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')


    return parser



def create_nerf():
    """Instantiate NeRF's MLP model.
    """
    parser = config_parser()
    args, unknown = parser.parse_known_args() #parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    width = 512
    scale = width ** -0.5

    rgba_mlp = NeRF(D=args.rgba_net_layers, W=args.rgba_net_width,
                        input_ch=width*2, output_ch=4) # RGBA

    clip_mlp = NeRF(D=args.clip_net_layers, W=args.clip_net_width,
                        input_ch=width, output_ch=width+1) # CLIP+Alpha 

    return args, rgba_mlp, clip_mlp


def raw2rgb(raw, z_vals, is_train=True):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)  # [N_rays, N_samples]

    #dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]


    alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    if not is_train:
        density = torch.sum(weights[...,None], -2).view(-1)
        weights[density!=0.] = weights[density!=0.] / density[density!=0.].view(-1,1)

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    #if not is_train:
    #    rgb_map[density==0.] = 0.5

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)


    return rgb_map, disp_map, acc_map, weights, depth_map


def raw2feature(raw, z_vals):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        feature_map: [num_rays, 512]. Estimated semantic feature of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)  # [N_rays, N_samples]

    #dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    #rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    feature = raw[...,:-1]

    alpha = raw2alpha(raw[...,-1], dists)  # [N_rays, N_samples]
    
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    feature_map = torch.sum(weights[...,None] * feature, -2)  # [N_rays, 3]
    feature_map = feature_map / torch.linalg.norm(feature_map, dim=-1, keepdim=True)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return feature_map, disp_map, acc_map, weights, depth_map



def get_rays(args, H, W):
    rel_y = np.expand_dims(np.linspace(args.near, args.far, args.N_samples),axis=0).repeat(H*W,axis=0)    
    fov_angle = np.deg2rad(args.camera_angle)
    half_W = W//2
    half_H = H//2
    tan_xy = np.array(([[i/half_W+1/W] for i in range(-half_W,half_W)])*W,np.float32) * math.tan(fov_angle/2.)
    rel_x = rel_y * tan_xy
    rel_z = rel_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,1)) * math.tan(fov_angle/2.))
    return (rel_x,rel_y,rel_z)


def RGB_to_BGR(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg

def run_nerf_rgb(args, model, scene_memory, position, direction, H=224, W=224, images=None, depths=None, is_train=True):

    camera_x, camera_y, camera_z = position
    heading_angle = - direction
    scene_fts, patch_directions, patch_scales, pcd, pcd_tree, fcd, fcd_tree, occupancy_pcd_tree = scene_memory
    patch_directions = torch.tensor(patch_directions + heading_angle, dtype=torch.float32).to("cuda")
    patch_directions = torch.cat((torch.sin(patch_directions).unsqueeze(-1), torch.cos(patch_directions).unsqueeze(-1)), dim=-1)
    patch_scales = torch.tensor(patch_scales, dtype=torch.float32).to("cuda").unsqueeze(-1)

    
    pcd_points = pcd[0].to("cuda")
    pcd_colors = torch.tensor(pcd[1]).to("cuda")

    fcd_points = fcd.to("cuda")

    rel_x, rel_y, rel_z = get_rays(args, H, W)
    if is_train:
        select_inds = np.random.choice(rel_x.shape[0], size=[args.N_rand], replace=False)
        rel_x = rel_x[select_inds]
        rel_y = rel_y[select_inds]
        rel_z = rel_z[select_inds]

    ray_x = rel_x * math.cos(heading_angle) + rel_y * math.sin(heading_angle) + camera_x
    ray_y = -rel_y * math.cos(heading_angle) + rel_x * math.sin(heading_angle) + camera_y
    ray_z = rel_z + camera_z
    ray_z_vals = rel_y


    ray_xyz = torch.tensor(np.concatenate((np.expand_dims(ray_x,-1),np.expand_dims(ray_y,-1),np.expand_dims(ray_z,-1)),axis=-1),dtype=torch.float32).to('cuda')

    occupancy_unit_length = 0.2**2
    with torch.no_grad():
        occupancy_query = ray_xyz.view(-1,3)
        searched_occupancy_dists, searched_occupancy_inds = occupancy_pcd_tree.query(occupancy_query, nr_nns_searches=1) #Note that the cupy_kdtree distances are squared
        occupancy_map = (searched_occupancy_dists < occupancy_unit_length).view(-1,)
        occupancy_ray_xyz = ray_xyz.view(-1,3)[occupancy_map]

        occupancy_ray_k_neighbor_dists, occupancy_ray_k_neighbor_inds = pcd_tree.query(occupancy_ray_xyz, nr_nns_searches=args.pointcloud_search_num)

    searched_ray_k_neighbor_dists = torch.full((ray_xyz.shape[0]*ray_xyz.shape[1],args.pointcloud_search_num),args.pointcloud_search_radius,dtype=torch.float32).to('cuda')
    searched_ray_k_neighbor_dists[occupancy_map] = occupancy_ray_k_neighbor_dists

    searched_ray_k_neighbor_inds = torch.full((ray_xyz.shape[0]*ray_xyz.shape[1],args.pointcloud_search_num),-1,dtype=torch.int64).to('cuda')
    searched_ray_k_neighbor_inds[occupancy_map] = occupancy_ray_k_neighbor_inds

    searched_ray_k_neighbor_dists = torch.sqrt(searched_ray_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
    searched_ray_k_neighbor_inds[searched_ray_k_neighbor_dists >= args.pointcloud_search_radius] = -1
    searched_ray_k_neighbor_dists[searched_ray_k_neighbor_dists >= args.pointcloud_search_radius] = args.pointcloud_search_radius


    searched_ray_k_neighbor_inds = searched_ray_k_neighbor_inds.view(ray_xyz.shape[0],ray_xyz.shape[1],args.pointcloud_search_num)
    searched_ray_k_neighbor_dists = searched_ray_k_neighbor_dists.view(ray_xyz.shape[0],ray_xyz.shape[1],args.pointcloud_search_num)

    sample_ray_xyz = torch.zeros((ray_xyz.shape[0],args.N_importance,3),dtype=torch.float32).to('cuda')
    sample_ray_z_vals = np.zeros((ray_xyz.shape[0],args.N_importance))

    for i in range(ray_xyz.shape[0]):
        idx = searched_ray_k_neighbor_inds[i]
        tmp_distance = searched_ray_k_neighbor_dists[i].sum(-1)
        tmp_density = (1/tmp_distance).cpu().numpy().tolist()

        peaks,_ = scipy.signal.find_peaks(tmp_density,distance=1)   
        topk = heapq.nlargest(args.N_importance, range(len(tmp_density)), tmp_density.__getitem__)
        k = max(args.N_importance//2, args.N_importance-len(peaks))
        topk_peaks = topk[:k]
        topk_peaks.extend(peaks[:args.N_importance-k])
        topk_peaks.sort()
        inds = np.array(topk_peaks,dtype=np.int64)
        sample_ray_xyz[i] = ray_xyz[i][torch.tensor(inds).to(ray_xyz.device)]
        sample_ray_z_vals[i] = ray_z_vals[i][inds]



    with torch.no_grad():
        sample_ray_k_neighbor_dists, sample_ray_k_neighbor_inds = pcd_tree.query(sample_ray_xyz.view(-1,3), nr_nns_searches=args.pointcloud_search_num)


    sample_ray_k_neighbor_dists = torch.sqrt(sample_ray_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
    sample_ray_k_neighbor_inds[sample_ray_k_neighbor_dists >= args.pointcloud_search_radius] = -1
    sample_ray_k_neighbor_dists[sample_ray_k_neighbor_dists >= args.pointcloud_search_radius] = args.pointcloud_search_radius
    sample_ray_k_neighbor_inds = sample_ray_k_neighbor_inds.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],args.pointcloud_search_num)
    sample_ray_k_neighbor_dists = sample_ray_k_neighbor_dists.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],args.pointcloud_search_num)



    with torch.no_grad():
        sample_feature_k_neighbor_dists, sample_feature_k_neighbor_inds = fcd_tree.query(sample_ray_xyz.view(-1,3), nr_nns_searches=args.featurecloud_search_num)

    sample_feature_k_neighbor_dists = torch.sqrt(sample_feature_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
    sample_feature_k_neighbor_inds[sample_feature_k_neighbor_dists >= args.featurecloud_search_radius] = -1
    sample_feature_k_neighbor_dists[sample_feature_k_neighbor_dists >= args.featurecloud_search_radius] = args.featurecloud_search_radius
    sample_feature_k_neighbor_inds = sample_feature_k_neighbor_inds.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],args.featurecloud_search_num)
    sample_feature_k_neighbor_dists = sample_feature_k_neighbor_dists.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],args.featurecloud_search_num)


    sample_ft_neighbor_xyzds = torch.zeros((sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],args.featurecloud_search_num,6),dtype=torch.float32).to('cuda')



    idx = sample_ray_k_neighbor_inds
    sample_ray_neighbor_xyz = pcd_points[idx] - sample_ray_xyz.unsqueeze(-2)
    sample_ray_neighbor_x = sample_ray_neighbor_xyz[...,0]
    sample_ray_neighbor_y = sample_ray_neighbor_xyz[...,1]

    # Get the relative angle to the NeRF camera, so the rotation angle is - heading_angle
    sample_ray_neighbor_xyz[...,0] = sample_ray_neighbor_x * math.cos(-heading_angle) + sample_ray_neighbor_y * math.sin(-heading_angle)
    sample_ray_neighbor_xyz[...,1] = -sample_ray_neighbor_y * math.cos(-heading_angle) + sample_ray_neighbor_x * math.sin(-heading_angle)

    sample_ray_neighbor_xyz[idx==-1] = args.far

    sample_ray_neighbor_rgb = pcd_colors[idx]
    sample_ray_neighbor_rgb[idx==-1] = 0

    idx = sample_feature_k_neighbor_inds 
    sample_ft_neighbor_xyzds[...,:3] = fcd_points[idx] - sample_ray_xyz.unsqueeze(-2)
    sample_ft_neighbor_x = sample_ft_neighbor_xyzds[...,0]
    sample_ft_neighbor_y = sample_ft_neighbor_xyzds[...,1]

    # Get the relative angle to the NeRF camera, so the rotation angle is - heading_angle
    sample_ft_neighbor_xyzds[...,0] = sample_ft_neighbor_x * math.cos(-heading_angle) + sample_ft_neighbor_y * math.sin(-heading_angle)
    sample_ft_neighbor_xyzds[...,1] = -sample_ft_neighbor_y * math.cos(-heading_angle) + sample_ft_neighbor_x * math.sin(-heading_angle)

    sample_ft_neighbor_xyzds[...,:3][idx==-1] = args.far
    sample_ft_neighbor_xyzds[...,3:5] = patch_directions[idx]
    sample_ft_neighbor_xyzds[...,3:5][idx==-1] = 0
    sample_ft_neighbor_xyzds[...,5:] = patch_scales[idx]
    sample_ft_neighbor_xyzds[...,5:][idx==-1] = 0


    sample_ft_neighbor_embedding = scene_fts[idx.cpu().numpy()]
    sample_ft_neighbor_embedding = torch.tensor(sample_ft_neighbor_embedding,dtype=torch.float32).to('cuda')
    sample_ft_neighbor_embedding[idx==-1] = 0



    if is_train:
        sample_ray_z_vals = torch.tensor(sample_ray_z_vals,dtype=torch.float32).to('cuda')

        sample_ray_neighbor_rgb = model.pcd_proj(sample_ray_neighbor_rgb)
        sample_ray_neighbor_xyz = model.pcd_position_embedding(sample_ray_neighbor_xyz)
        sample_ray = model.pcd_aggregation( (sample_ray_neighbor_rgb + sample_ray_neighbor_xyz).view(-1,args.N_importance, args.pointcloud_search_num*128) )


        sample_ft_neighbor_xyzds = model.fcd_position_embedding(sample_ft_neighbor_xyzds)
        sample_ft = model.fcd_aggregation( (sample_ft_neighbor_embedding + sample_ft_neighbor_xyzds).view(-1,args.N_importance, args.featurecloud_search_num*512) )

        sample_input = torch.cat((sample_ray,sample_ft),-1)
        sample_rgba = model.rgba_mlp(sample_input.view(-1,sample_input.shape[-1])).view(-1,args.N_importance,4)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2rgb(sample_rgba, sample_ray_z_vals.view(-1,args.N_importance))

        rgb_target = images.reshape((-1,3))[select_inds]
        rgb_target = torch.tensor(rgb_target.astype(np.float32)).to(rgb_map.device) / 255.
        img_loss = img2mse(rgb_map, rgb_target)

        if torch.any(torch.isnan(img_loss)):
            print("RGB has NaN!!!!!!!!!!!")

        return  img_loss  #/ args.N_rand

    else:      
        with torch.no_grad():
            image = np.zeros((H*W,3), dtype=np.uint8)
            depth = np.zeros((H*W), dtype=np.float32)

            for i in range(math.ceil(sample_ray_neighbor_xyz.shape[0]/args.chunk)):
                sample_ray_z_vals_chunk = torch.tensor(sample_ray_z_vals[i*args.chunk:(i+1)*args.chunk],dtype=torch.float32).to('cuda')
                sample_ray_neighbor_rgb_chunk = sample_ray_neighbor_rgb[i*args.chunk:(i+1)*args.chunk].to('cuda')

                sample_ray_neighbor_rgb_chunk = model.pcd_proj(sample_ray_neighbor_rgb_chunk)
                sample_ray_neighbor_xyz_chunk = model.pcd_position_embedding(sample_ray_neighbor_xyz[i*args.chunk:(i+1)*args.chunk])
                sample_ray_chunk = model.pcd_aggregation( (sample_ray_neighbor_rgb_chunk + sample_ray_neighbor_xyz_chunk).view(-1,args.N_importance, args.pointcloud_search_num*128) )


                sample_ft_neighbor_embedding_chunk = sample_ft_neighbor_embedding[i*args.chunk:(i+1)*args.chunk].to('cuda')

                sample_ft_neighbor_xyzds_chunk = model.fcd_position_embedding(sample_ft_neighbor_xyzds[i*args.chunk:(i+1)*args.chunk])
                sample_ft_chunk = model.fcd_aggregation( (sample_ft_neighbor_embedding_chunk + sample_ft_neighbor_xyzds_chunk).view(-1,args.N_importance, args.featurecloud_search_num*512) )

                sample_input_chunk = torch.cat((sample_ray_chunk,sample_ft_chunk),-1)
                sample_rgba_chunk = model.rgba_mlp(sample_input_chunk.view(-1,sample_input_chunk.shape[-1])).view(-1,args.N_importance,4)

                rgb_map_chunk, disp_map_chunk, acc_map_chunk, weights_chunk, depth_map_chunk = raw2rgb(sample_rgba_chunk, sample_ray_z_vals_chunk.view(-1,args.N_importance),is_train)

                image[i*args.chunk:(i+1)*args.chunk] = (rgb_map_chunk * 255.).cpu().numpy().astype(np.uint8)
                depth[i*args.chunk:(i+1)*args.chunk] = depth_map_chunk.cpu().numpy().astype(np.float32)
            #cv2.imwrite('test.jpg', RGB_to_BGR(image.reshape((224,224,3))))
            #exit()
            return image.reshape((H,W,3)), depth.reshape((H,W))


def run_nerf_feature(args, model, scene_memory, position, direction, H=8, W=8, clip_fts=None, is_train=True):

    target_fts, target_patch_id = clip_fts
    camera_x, camera_y, camera_z = position
    heading_angle = - direction
    scene_fts, patch_directions, patch_scales, _, _, fcd, fcd_tree, occupancy_pcd_tree = scene_memory
    patch_directions = torch.tensor(patch_directions + heading_angle, dtype=torch.float32).to("cuda")
    patch_directions = torch.cat((torch.sin(patch_directions).unsqueeze(-1), torch.cos(patch_directions).unsqueeze(-1)), dim=-1)
    patch_scales = torch.tensor(patch_scales, dtype=torch.float32).to("cuda").unsqueeze(-1)

    fcd_points = fcd.to("cuda")

    rel_x, rel_y, rel_z = get_rays(args, H, W)
  
    ray_x = rel_x * math.cos(heading_angle) + rel_y * math.sin(heading_angle) + camera_x
    ray_y = -rel_y * math.cos(heading_angle) + rel_x * math.sin(heading_angle) + camera_y
    ray_z = rel_z + camera_z
    ray_z_vals = rel_y


    ray_xyz = torch.tensor(np.concatenate((np.expand_dims(ray_x,-1),np.expand_dims(ray_y,-1),np.expand_dims(ray_z,-1)),axis=-1),dtype=torch.float32).to('cuda')

    occupancy_unit_length = 1**2
    with torch.no_grad():
        occupancy_query = ray_xyz.view(-1,3)
        searched_occupancy_dists, searched_occupancy_inds = occupancy_pcd_tree.query(occupancy_query, nr_nns_searches=1) #Note that the cupy_kdtree distances are squared
        occupancy_map = (searched_occupancy_dists < occupancy_unit_length).view(-1,)
        occupancy_ray_xyz = ray_xyz.view(-1,3)[occupancy_map]

        occupancy_ray_k_neighbor_dists, occupancy_ray_k_neighbor_inds = fcd_tree.query(occupancy_ray_xyz, nr_nns_searches=args.featurecloud_search_num)

    searched_ray_k_neighbor_dists = torch.full((ray_xyz.shape[0]*ray_xyz.shape[1],args.featurecloud_search_num),args.featurecloud_search_radius,dtype=torch.float32).to('cuda')
    searched_ray_k_neighbor_dists[occupancy_map] = occupancy_ray_k_neighbor_dists

    searched_ray_k_neighbor_inds = torch.full((ray_xyz.shape[0]*ray_xyz.shape[1],args.featurecloud_search_num),-1,dtype=torch.int64).to('cuda')
    searched_ray_k_neighbor_inds[occupancy_map] = occupancy_ray_k_neighbor_inds

    searched_ray_k_neighbor_dists = torch.sqrt(searched_ray_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
    searched_ray_k_neighbor_inds[searched_ray_k_neighbor_dists >= args.featurecloud_search_radius] = -1
    searched_ray_k_neighbor_dists[searched_ray_k_neighbor_dists >= args.featurecloud_search_radius] = args.featurecloud_search_radius


    searched_ray_k_neighbor_inds = searched_ray_k_neighbor_inds.view(ray_xyz.shape[0],ray_xyz.shape[1],args.featurecloud_search_num)
    searched_ray_k_neighbor_dists = searched_ray_k_neighbor_dists.view(ray_xyz.shape[0],ray_xyz.shape[1],args.featurecloud_search_num)

    sample_ray_xyz = torch.zeros((ray_xyz.shape[0],args.N_importance,3),dtype=torch.float32).to('cuda')
    sample_ray_z_vals = np.zeros((ray_xyz.shape[0],args.N_importance))

    for i in range(ray_xyz.shape[0]):
        idx = searched_ray_k_neighbor_inds[i]
        tmp_distance = searched_ray_k_neighbor_dists[i].sum(-1)
        tmp_density = (1/tmp_distance).cpu().numpy().tolist()

        peaks,_ = scipy.signal.find_peaks(tmp_density,distance=1)   
        topk = heapq.nlargest(args.N_importance, range(len(tmp_density)), tmp_density.__getitem__)
        k = max(args.N_importance//2, args.N_importance-len(peaks))
        topk_peaks = topk[:k]
        topk_peaks.extend(peaks[:args.N_importance-k])
        topk_peaks.sort()
        inds = np.array(topk_peaks,dtype=np.int64)
        sample_ray_xyz[i] = ray_xyz[i][torch.tensor(inds).to(ray_xyz.device)]
        sample_ray_z_vals[i] = ray_z_vals[i][inds]


    with torch.no_grad():
        sample_feature_k_neighbor_dists, sample_feature_k_neighbor_inds = fcd_tree.query(sample_ray_xyz.view(-1,3), nr_nns_searches=args.featurecloud_search_num)

    sample_feature_k_neighbor_dists = torch.sqrt(sample_feature_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
    sample_feature_k_neighbor_inds[sample_feature_k_neighbor_dists >= args.featurecloud_search_radius] = -1
    sample_feature_k_neighbor_dists[sample_feature_k_neighbor_dists >= args.featurecloud_search_radius] = args.featurecloud_search_radius
    sample_feature_k_neighbor_inds = sample_feature_k_neighbor_inds.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],args.featurecloud_search_num)
    sample_feature_k_neighbor_dists = sample_feature_k_neighbor_dists.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],args.featurecloud_search_num)


    sample_ft_neighbor_xyzds = torch.zeros((sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],args.featurecloud_search_num,6),dtype=torch.float32).to('cuda')

    idx = sample_feature_k_neighbor_inds 
    sample_ft_neighbor_xyzds[...,:3] = fcd_points[idx] - sample_ray_xyz.unsqueeze(-2)

    sample_ft_neighbor_x = sample_ft_neighbor_xyzds[...,0]
    sample_ft_neighbor_y = sample_ft_neighbor_xyzds[...,1]
    # Get the relative angle to the NeRF camera, so the rotation angle is - heading_angle
    sample_ft_neighbor_xyzds[...,0] = sample_ft_neighbor_x * math.cos(-heading_angle) + sample_ft_neighbor_y * math.sin(-heading_angle)
    sample_ft_neighbor_xyzds[...,1] = -sample_ft_neighbor_y * math.cos(-heading_angle) + sample_ft_neighbor_x * math.sin(-heading_angle)

    sample_ft_neighbor_xyzds[...,:3][idx==-1] = args.far
    sample_ft_neighbor_xyzds[...,3:5] = patch_directions[idx]
    sample_ft_neighbor_xyzds[...,3:5][idx==-1] = 0
    sample_ft_neighbor_xyzds[...,5:] = patch_scales[idx]
    sample_ft_neighbor_xyzds[...,5:][idx==-1] = 0


    sample_ft_neighbor_embedding = scene_fts[idx.cpu().numpy()]
    sample_ft_neighbor_embedding = torch.tensor(sample_ft_neighbor_embedding,dtype=torch.float32).to('cuda')
    sample_ft_neighbor_embedding[idx==-1] = 0


    sample_ft_neighbor_xyzds = model.fcd_position_embedding(sample_ft_neighbor_xyzds)
    sample_ft = model.fcd_aggregation( (sample_ft_neighbor_embedding + sample_ft_neighbor_xyzds).view(-1,args.N_importance, args.featurecloud_search_num*512) )

    sample_input = sample_ft

    sample_feature = model.clip_mlp(sample_input.view(-1,sample_input.shape[-1])).view(-1,args.N_importance,512+1)

    sample_ray_z_vals = torch.tensor(sample_ray_z_vals,dtype=torch.float32).to('cuda')
    feature_map, disp_map, acc_map, weights, depth_map = raw2feature(sample_feature, sample_ray_z_vals.view(-1,args.N_importance))
    
    transformer_input = torch.cat((model.class_embedding,feature_map),dim=0)
    transformer_input = transformer_input + model.positional_embedding

    predicted_fts = model.nerf_view_encoder(transformer_input.unsqueeze(0)).squeeze(0)


    if is_train:

        entire_fts = predicted_fts[0:1]
        subregion_fts = predicted_fts[1:][target_patch_id]
        predicted_fts = torch.cat([entire_fts,subregion_fts],dim=0)

        # cosine similarity
        predicted_fts = predicted_fts / torch.linalg.norm(predicted_fts, dim=-1, keepdim=True)
        target_fts = target_fts / torch.linalg.norm(target_fts, dim=-1, keepdim=True)

        ft_loss = (1. - (predicted_fts * target_fts).sum(-1)).mean() * args.feature_loss_weight
        if torch.any(torch.isnan(ft_loss)):
            print("Feature has NaN!!!!!!!!!!!")
            return 0. * (model.clip_mlp(torch.zeros((1,512),dtype=torch.float32,device='cuda')).mean() + model.nerf_view_encoder(torch.zeros((1,2,512),dtype=torch.float32,device='cuda')).mean() + model.class_embedding.mean() + model.positional_embedding.mean())

        return ft_loss
    else:
        return predicted_fts



        