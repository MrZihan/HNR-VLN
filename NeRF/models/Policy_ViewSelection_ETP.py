from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from NeRF.models.etp.vlnbert_init import get_vlnbert_models
from NeRF.common.aux_losses import AuxLosses
from NeRF.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from NeRF.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from NeRF.models.policy import ILPolicy

from NeRF.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from NeRF.waypoint_pred.utils import nms
from NeRF.models.graph_utils import  MAX_DIST
from NeRF.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, length2mask, angle_feature_torch, pad_tensors, gen_seq_masks, get_angle_fts, get_angle_feature, get_point_angle_feature, calculate_vp_rel_pos_fts, calc_position_distance,pad_tensors_wgrad, rectangular_to_polar)
import math
from PIL import Image
import cv2
import open3d as o3d
from torch_kdtree import build_kd_tree
import numpy as np
image_global_x_db = []
image_global_y_db = []
image_db=[]
DATASET = 'R2R'
RGB_HW = 112
global_count = 0

@baseline_registry.register_policy
class PolicyViewSelectionETP(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            ETP(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class ETP(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the ETP model ...')
        self.vln_bert = get_vlnbert_models(config=model_config)

        self.drop_env = nn.Dropout(p=0.4)


        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))

        # Init the RGB encoder
        # assert model_config.RGB_ENCODER.cnn_type in [
        #     "TorchVisionResNet152", "TorchVisionResNet50"
        # ], "RGB_ENCODER.cnn_type must be TorchVisionResNet152 or TorchVisionResNet50"
        # if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
        #     self.rgb_encoder = TorchVisionResNet50(
        #         observation_space,
        #         model_config.RGB_ENCODER.output_size,
        #         device,
        #         spatial_output=model_config.spatial_output,
        #     )
        self.rgb_encoder = CLIPEncoder(self.device,16)
        #self.grid_encoder = CLIPEncoder(self.device,16)
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
    
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # 逆时针
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # 对应到逆时针
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))


        batch_size = self.pano_angle_fts.shape[0]
        self.global_fts = [[] for i in range(batch_size)]
        self.global_position_x = [[] for i in range(batch_size)]
        self.global_position_y = [[] for i in range(batch_size)]
        self.global_position_z = [[] for i in range(batch_size)]
        self.global_patch_scales = [[] for i in range(batch_size)]
        self.global_patch_directions = [[] for i in range(batch_size)]
        self.global_mask = [[] for i in range(batch_size)]
        self.max_x = [-10000 for i in range(batch_size)]
        self.min_x = [10000 for i in range(batch_size)]
        self.max_y = [-10000 for i in range(batch_size)]
        self.min_y = [10000 for i in range(batch_size)]
        self.headings = [0 for i in range(batch_size)]
        self.positions = None
        self.global_map_index = [[] for i in range(batch_size)]

        self.pointcloud_x = [[] for i in range(batch_size)]
        self.pointcloud_y = [[] for i in range(batch_size)]
        self.pointcloud_z = [[] for i in range(batch_size)]
        self.pointcloud_rgb = [[] for i in range(batch_size)]

        self.pointcloud_fcd = [[] for i in range(batch_size)]
        self.pointcloud_pcd = [[] for i in range(batch_size)]

        self.start_positions = None
        self.start_headings = None
        self.action_step = 0
        self.train()

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def preprocess_depth(self, depth):
        # depth - (B, H, W, 1) torch Tensor
        global DATASET
        if DATASET == 'R2R':
            min_depth = 0.
            max_depth = 10.
        elif DATASET == 'RxR':
            min_depth = 0.5
            max_depth = 5.0

        # Column-wise post-processing
        depth = depth * 1.0
        H = depth.shape[1]
        depth_max, _ = depth.max(dim=1, keepdim=True)  # (B, H, W, 1)
        depth_max = depth_max.expand(-1, H, -1, -1)
        depth[depth == 0] = depth_max[depth == 0]

        #mask2 = depth > 0.99
        #depth[mask2] = 0 # noise

        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        depth = depth / 100.
        return depth

    def forward(self, mode=None, 
                txt_ids=None, txt_masks=None, txt_embeds=None, 
                waypoint_predictor=None, observations=None, in_train=True,
                rgb_fts=None, dep_fts=None, loc_fts=None, 
                nav_types=None, view_lens=None,
                gmap_vp_ids=None, gmap_step_ids=None,
                gmap_img_fts=None, gmap_pos_fts=None, 
                gmap_masks=None, gmap_visited_masks=None, gmap_pair_dists=None):

        global DATASET, RGB_HW
        if mode == 'language':
            encoded_sentence = self.vln_bert.forward_txt(
                txt_ids, txt_masks,
            )
            return encoded_sentence

        elif mode == 'waypoint':
            #batch_size = observations['instruction'].size(0)
            batch_size = observations['rgb'].shape[0]

            ''' encoding rgb/depth at all directions ----------------------------- '''
            NUM_ANGLES = 120    # 120 angles 3 degrees each
            NUM_IMGS = 12
            NUM_CLASSES = 12    # 12 distances at each sector
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            # reverse the order of input images to clockwise
            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if 'depth' in k:  # You might need to double check the keys order
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                        depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi] 
                    a_count += 1

            global global_count
            #for i in range(12):
            #    cv2.imwrite('%d_%d_origin.jpg'%(global_count,i), self.RGB_to_BGR(rgb_batch[i].cpu().numpy()))
            #    cv2.imwrite('%d_%d_depth.jpg'%(global_count,i), (depth_batch[i].cpu().numpy() * 255).astype(np.uint8))
            global_count += 1

            obs_view12 = {}
            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch

            depth_batch = torch.zeros((12,RGB_HW,RGB_HW,1))
            depth_batch_fts = torch.zeros((12,14,14,1))
            for i in range(obs_view12['depth'].shape[0]):
                depth_batch[i] = torch.tensor(cv2.resize(obs_view12['depth'][i].cpu().numpy(), (RGB_HW, RGB_HW),  interpolation = cv2.INTER_NEAREST)).view(RGB_HW,RGB_HW,1)
                depth_batch_fts[i] = torch.tensor(cv2.resize(obs_view12['depth'][i].cpu().numpy(), (14, 14),  interpolation = cv2.INTER_NEAREST)).view(14,14,1)


            depth_batch = self.preprocess_depth(depth_batch)
            depth_batch_fts = self.preprocess_depth(depth_batch_fts).numpy()

            image_list = [Image.fromarray(rgb_batch[img_id].cpu().numpy()).resize((RGB_HW, RGB_HW),Image.ANTIALIAS) for img_id in range(rgb_batch.shape[0])]

            depth_batch = depth_batch.view(batch_size,12,depth_batch.shape[-3],depth_batch.shape[-2]).cpu().numpy()

            with torch.no_grad():
                depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])
                #grid_fts = self.grid_encoder(obs_view12, fine_grained_fts=True)      # torch.Size([bs, 2048, 7, 7])
                grid_fts = self.rgb_encoder(obs_view12, fine_grained_fts=True)      # torch.Size([bs, 2048, 7, 7])

            rgb_embedding = grid_fts[:,0]

            batch_grid_fts = grid_fts.view(batch_size,12,197,512).cpu().numpy()

           
            
            for i in range(batch_size):
                position = {}
                position['x'] = self.positions[i][0]
                position['y'] = self.positions[i][-1]
                position['z'] = self.positions[i][1]
                heading = self.headings[i]
                depth = depth_batch[i]
                grid_ft = batch_grid_fts[i]
                self.getGlobalMap(i, position, heading, depth, depth_batch_fts, grid_ft,image_list)


            ''' waypoint prediction ----------------------------- '''
            waypoint_heatmap_logits = waypoint_predictor(
                rgb_embedding, depth_embedding)

            # reverse the order of images back to counter-clockwise
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, NUM_IMGS, 512, 1, 1)
            depth_embed_reshape = depth_embedding.reshape(
                batch_size, NUM_IMGS, 128, 4, 4)
            rgb_feats = torch.cat((
                rgb_embed_reshape[:,0:1,:], 
                torch.flip(rgb_embed_reshape[:,1:,:], [1]),
            ), dim=1)
            depth_feats = torch.cat((
                depth_embed_reshape[:,0:1,:], 
                torch.flip(depth_embed_reshape[:,1:,:], [1]),
            ), dim=1)
            # way_feats = torch.cat((
            #     way_feats[:,0:1,:], 
            #     torch.flip(way_feats[:,1:,:], [1]),
            # ), dim=1)

            # from heatmap to points
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(
                    batch_size, NUM_ANGLES*NUM_CLASSES,
                ), dim=1
            )
            batch_x_norm = batch_x_norm.reshape(
                batch_size, NUM_ANGLES, NUM_CLASSES,
            )
            batch_x_norm_wrap = torch.cat((
                batch_x_norm[:,-1:,:], 
                batch_x_norm, 
                batch_x_norm[:,:1,:]), 
                dim=1)
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), 
                max_predictions=5,
                sigma=(7.0,5.0))

            # predicted waypoints before sampling
            batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

            # candidate_lengths = ((batch_output_map!=0).sum(-1).sum(-1) + 1).tolist()
            # if isinstance(candidate_lengths, int):
            #     candidate_lengths = [candidate_lengths]
            # max_candidate = max(candidate_lengths)  # including stop
            # cand_mask = length2mask(candidate_lengths, device=self.device)

            if in_train:
                # Waypoint augmentation
                # parts of heatmap for sampling (fix offset first)
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (waypoint_heatmap_logits[:,-HEATMAP_OFFSET:,:], 
                    waypoint_heatmap_logits[:,:-HEATMAP_OFFSET,:],
                ), dim=1)
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                # batch_way_log_prob = []
                for j in range(batch_size):
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # clockwise image indexes (same as batch_x_norm)
                    img_idxes = ((angle_idxes.cpu().numpy()+5) // 10)
                    img_idxes[img_idxes==12] = 0
                    # # candidate waypoint states
                    # way_feats_regional = way_feats[j][img_idxes]
                    # heatmap regions for sampling
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(img_idxes.size, -1)
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()
                    sample_angle_idxes = []
                    sample_distance_idxes = []
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(torch.div(way_act, 12, rounding_mode='floor')+angle_pointer)
                        sample_distance_idxes.append(way_act%12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)
                    # batch_way_log_prob.append(
                    #     probs_c.log_prob(way_heats_act))
            else:
                # batch_way_log_prob = None
                None
            
            rgb_feats = self.space_pool_rgb(rgb_feats)
            depth_feats = self.space_pool_depth(depth_feats)

            # for cand
            cand_rgb = []
            cand_depth = []
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []
            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                # for angle & distance
                angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi       # 顺时针
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # 逆时针
                cand_angle_fts.append( angle_feature_torch(angle_rad_c) )
                cand_angles.append(angle_rad_cc.tolist())
                cand_distances.append( ((distance_idxes + 1)*0.25).tolist() )
                # for img idxes
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # 逆时针
                img_idxes[img_idxes==12] = 0
                cand_img_idxes.append(img_idxes)
                # for rgb & depth
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                cand_depth.append(depth_feats[j, img_idxes, ...])
            
            # for pano
            pano_rgb = rgb_feats                            # B x 12 x 2048
            pano_depth = depth_feats                        # B x 12 x 128
            pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
            pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

            # cand_angle_fts 顺时针
            # cand_angles 逆时针

            batch_grid_fts  = [torch.tensor(self.global_fts[index]).cuda() for index in range(batch_size)]
            batch_map_index  = [torch.tensor(self.global_map_index[index]).cuda() for index in range(batch_size)]
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 2048]
                'cand_depth': cand_depth,           # [K x 128]
                'cand_angle_fts': cand_angle_fts,   # [K x 4]
                'cand_img_idxes': cand_img_idxes,   # [K]
                'cand_angles': cand_angles,         # [K]
                'cand_distances': cand_distances,   # [K]

                'pano_rgb': pano_rgb,               # B x 12 x 2048
                'pano_depth': pano_depth,           # B x 12 x 128
                'pano_angle_fts': pano_angle_fts,   # 12 x 4
                'pano_img_idxes': pano_img_idxes,   # 12 
                'batch_grid_fts': batch_grid_fts, 
            }
            
            return outputs

        elif mode == 'panorama':
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(
                rgb_fts, dep_fts, loc_fts, nav_types, view_lens,
            )
            return outs

        elif mode == 'navigation':
            outs = self.vln_bert.forward_navigation(
                txt_embeds, txt_masks, 
                gmap_vp_ids, gmap_step_ids,
                gmap_img_fts, gmap_pos_fts, 
                gmap_masks, gmap_visited_masks, gmap_pair_dists, batch_grid_fts, None, None
            )
            return outs


    def get_rel_position(self,depth_map,angle):
        global DATASET
        W=14
        H=14
        half_W = W//2
        half_H = H//2
        depth_y = depth_map.astype(np.float32) # / 4000.
        if DATASET == 'R2R':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi/4.)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi/4.))
            scale = depth_y * math.tan(math.pi/4.) * 2. / W

        elif DATASET == 'RxR':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi * 79./360.)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi * 79./360.))
            scale = depth_y * math.tan(math.pi * 79./360.) * 2. / W

        direction = (direction+angle) % (2*math.pi)
        rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
        rel_y = -depth_y * math.cos(angle) + depth_x * math.sin(angle)
        rel_z = depth_z
        return rel_x, rel_y, rel_z, direction.reshape(-1), scale.reshape(-1)

    def image_get_rel_position(self,depth_map,angle):
        global DATASET, RGB_HW
        W=RGB_HW
        H=RGB_HW
        half_W = W//2
        half_H = H//2
        depth_y = depth_map.astype(np.float32)
        if DATASET == 'R2R':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi/4)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi/4.))
        elif DATASET == 'RxR':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi * 79./360.)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi * 79./360.))

        direction = (direction+angle) % (2*math.pi)
        rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
        rel_y = -depth_y * math.cos(angle) + depth_x * math.sin(angle)
        rel_z = depth_z

        return rel_x, rel_y, rel_z, direction.reshape(-1)

    def RGB_to_BGR(self,cvimg):
        pilimg = cvimg.copy()
        pilimg[:, :, 0] = cvimg[:, :, 2]
        pilimg[:, :, 2] = cvimg[:, :, 0]
        return pilimg



    def getSceneMemory(self, batch_id, only_feature=False):

        i = batch_id
        #angle = - heading + math.pi
        
        # FeatureCloud
        global_position_x = np.concatenate(self.global_position_x[i],0)
        global_position_y = np.concatenate(self.global_position_y[i],0)
        global_position_z = np.concatenate(self.global_position_z[i],0)
        global_patch_scales = np.concatenate(self.global_patch_scales[i],0)
        global_patch_directions = np.concatenate(self.global_patch_directions[i],0)

        
        map_x = global_position_x
        map_y = global_position_y
        map_z = global_position_z

        fcd =  torch.tensor(np.concatenate((map_x.reshape((-1,1)),map_y.reshape((-1,1)),map_z.reshape((-1,1))),axis=-1),dtype=torch.float32).to("cuda")
        fcd_tree = build_kd_tree(fcd)


        global_position_x = global_position_x.reshape((-1,))
        global_position_y = global_position_y.reshape((-1,))
        global_position_z = global_position_z.reshape((-1,))
        
        global_fts = self.global_fts[i]

        #heading_angles = rectangular_to_polar(global_position_x,global_position_y)
        
        #select_ids = (global_position_y>0.) & (np.abs(np.degrees(heading_angles)) <= angular_range/2.) & (np.abs(global_patch_directions)<(math.pi/3.*2.))

        selected_fts = global_fts#[select_ids]
        selected_patch_scales = global_patch_scales#[select_ids]
        selected_patch_directions = global_patch_directions#[select_ids]

        selected_position_x = global_position_x#[select_ids]
        selected_position_y = global_position_y#[select_ids]
        selected_position_z = global_position_z#[select_ids]

  
        # PointCloud
        if only_feature:
            # Featurecloud Occupancy Map
            occupancy_pcd = torch.div(fcd,0.1, rounding_mode='floor')
            occupancy_pcd = torch.unique(occupancy_pcd, dim=0) * 0.1
            occupancy_pcd_tree = build_kd_tree(occupancy_pcd)
            pcd_tree = None
            pcd = None
        else:
            map_x = np.concatenate(self.pointcloud_x[i],axis=-1)
            map_y = np.concatenate(self.pointcloud_y[i],axis=-1)
            map_z = np.concatenate(self.pointcloud_z[i],axis=-1)

            map_rgb = np.concatenate([img.reshape((-1,3)) for img in self.pointcloud_rgb[i]],axis=0).astype(np.float32) / 255.

            pcd =  [torch.tensor(np.concatenate((map_x.reshape((-1,1)),map_y.reshape((-1,1)),map_z.reshape((-1,1))),axis=-1),dtype=torch.float32).to('cuda'),map_rgb]
            pcd_tree = build_kd_tree(pcd[0])


            # Pointcloud Occupancy Map
            occupancy_pcd = torch.div(pcd[0], 0.1, rounding_mode='floor')
            occupancy_pcd = torch.unique(occupancy_pcd, dim=0) * 0.1
            occupancy_pcd_tree = build_kd_tree(occupancy_pcd)

        
        # Pointcloud Visualize
        '''
        step_num = len(self.global_mask[i])
        pcd_visulize = o3d.geometry.PointCloud()
        points_data =  np.concatenate((map_x.reshape((-1,1)),-map_y.reshape((-1,1)),map_z.reshape((-1,1))),axis=-1) # Be careful, it's "-map_y" !!!!!!
        pcd_visulize.points = o3d.utility.Vector3dVector(points_data)
        pcd_visulize.colors = o3d.utility.Vector3dVector(map_rgb)
        o3d.io.write_point_cloud('%d_%d.xyzrgb' % (i, step_num), pcd_visulize)
        '''

        return selected_fts, (selected_position_x, selected_position_y, selected_position_z), selected_patch_directions, selected_patch_scales, pcd, fcd, pcd_tree, fcd_tree, occupancy_pcd_tree


    def getGlobalMap(self, batch_id, position, heading, depth, depth_batch_fts, grid_ft,image_list):
        global image_global_x_db, image_global_y_db, image_db
        global DATASET
        GLOBAL_WIDTH = 14
        GLOBAL_HEIGHT = 14
        i = batch_id
        viewpoint_x_list = []
        viewpoint_y_list = []
        viewpoint_z_list = []
        viewpoint_scale_list = []
        viewpoint_direction_list = []
        
    
        for ix in range(12):
            image = image_list[ix]
            rel_x, rel_y, rel_z, direction = self.image_get_rel_position(depth[ix:ix+1].reshape(-1),ix*math.pi/6-self.headings[i])  

            image_global_x = rel_x + position["x"]
            image_global_y = rel_y + position["y"]  
            image_global_z = rel_z + position["z"]  
            self.pointcloud_x[i].append(image_global_x)
            self.pointcloud_y[i].append(image_global_y)
            self.pointcloud_z[i].append(image_global_z)
            image = np.array(image).reshape(-1,3)
            self.pointcloud_rgb[i].append(image)

        '''
        # Top-down Map Visulize
        test_image = np.zeros((512, 512, 3), dtype=np.uint8)
        step_num = len(self.global_mask[i])
        for item in range(len(self.pointcloud_rgb[i])):
            angle = -self.headings[i] + math.pi
            tmp_x = self.pointcloud_x[i][item] - position["x"]
            tmp_y = self.pointcloud_y[i][item] - position["y"]
            
            map_z = self.pointcloud_z[i][item] - position["z"]
            map_x = -(tmp_x * math.cos(angle) + tmp_y * math.sin(angle))
            map_y = tmp_y * math.cos(angle) - tmp_x * math.sin(angle)
            map_x = map_x[map_z<0.5]
            map_y = map_y[map_z<0.5]
            test_image[((map_x+10)/(20)*511).astype(np.int32),((map_y+10)/(20)*511).astype(np.int32),:] = self.pointcloud_rgb[i][item][map_z<0.5]

        test_image[252:258,252:258,0] = 0
        test_image[252:258,252:258,1] = 0
        test_image[252:258,252:258,2] = 255
        cv2.imwrite('top_down_%d_%d.jpg' % (i, step_num), self.RGB_to_BGR(test_image))
        '''

        depth = depth_batch_fts.reshape((12,-1))
       
        for ix in range(12):
            rel_x, rel_y, rel_z, direction, scale = self.get_rel_position(depth[ix:ix+1],ix*math.pi/6-self.headings[i])  
            global_x = rel_x + position["x"]
            global_y = rel_y + position["y"]
            global_z = rel_z + position["z"]

            viewpoint_x_list.append(global_x)
            viewpoint_y_list.append(global_y)
            viewpoint_z_list.append(global_z)
            viewpoint_scale_list.append(scale)
            viewpoint_direction_list.append(direction)

        
        if self.global_fts[i] == []:
            self.global_fts[i] = grid_ft[:,1:].reshape((-1,512))
            
        else:
            self.global_fts[i] = np.concatenate((self.global_fts[i],grid_ft[:,1:].reshape((-1,512))),axis=0)


        position_x = np.concatenate(viewpoint_x_list,0)
        position_y = np.concatenate(viewpoint_y_list,0)
        position_z = np.concatenate(viewpoint_z_list,0)
        patch_scales = np.concatenate(viewpoint_scale_list,0)
        patch_directions = np.concatenate(viewpoint_direction_list,0)
        self.global_position_x[i].append(position_x)
        self.global_position_y[i].append(position_y)
        self.global_position_z[i].append(position_z)
        self.global_patch_scales[i].append(patch_scales)
        self.global_patch_directions[i].append(patch_directions)



class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

