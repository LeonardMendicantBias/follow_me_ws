from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18 as resnet, ResNet18_Weights
from torchvision import transforms

from sklearn.linear_model import Ridge

import cv2
import numpy as np


class Tracker:
	"""
		Tracker that use Kalman filter and ResNet18 for re-identification.
	"""
	part_ids = torch.tensor([1, 5, 6, 11, 12, 13, 14, 15, 16], dtype=torch.int)
	part_names = [
		"head", "neck", "torso", "left_shoulder", "right_shoulder",
		"left_elbow", "right_elbow", "left_hand", "right_hand"
	]
	part_masks = torch.tensor([
		[1, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 1, 1, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 1, 1, 1, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 1, 1],
	], dtype=torch.int)
	N = part_masks.shape[0]

	def __init__(self,
		img_size,
		device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	):
		self.img_size = img_size
		self.device = device
		self.conf_thres = 0.5

		self._has_target = False
		self.memory_size = 25
		self.min_pos_example = 1
		
		self.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((img_size[0], img_size[1])),  # (h, w) 
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])
		self.reset()

	def reset(self):
		model = resnet(weights=ResNet18_Weights.IMAGENET1K_V1)
		self.resnet = nn.Sequential(*list(model.children())[:-2])
		self.resnet.eval()
		self.resnet.to(self.device)

		self._has_target = False
		self.lt_pos_memory = [[] for _ in range(self.N+1)]
		self.st_pos_memory = [[] for _ in range(self.N+1)]
		self.lt_neg_memory = [[] for _ in range(self.N+1)]
		self.st_neg_memory = [[] for _ in range(self.N+1)]
		self.classifiers = [Ridge(alpha=1.0) for _ in range(self.N+1)]

	def process_crop(self,
		rgb_img: torch.Tensor,
		bboxes: List[Dict[str, Union[int, float]]],
		kpts, 
		confs
	):
		bboxes = torch.tensor(bboxes).to(self.device)
		kpts = torch.tensor(kpts).to(self.device)
		confs = torch.tensor(confs).to(self.device)
		
		crop_imgs = [
			rgb_img[
				int(bbox[1]-bbox[3]//2):int(bbox[1]+bbox[3]//2), int(bbox[0]-bbox[2]//2):int(bbox[0]+bbox[2]//2)
			] for bbox in bboxes
		]
		bbox_features = self.resnet(
			torch.stack([
				self.transform(crop_img) for crop_img in crop_imgs
			]).to(self.device)
		)  # (K, D, H, W)
		# print("bbox_features", bbox_features.shape)
		K = kpts.shape[0]

		# (K, 17) -> (K, 9) -> (K, N, 9)
		visible_kpt = (confs[:, self.part_ids].unsqueeze(1) * self.part_masks.to(self.device).unsqueeze(0))
		visible_kpt = visible_kpt > self.conf_thres
		visible_part = visible_kpt.amax(-1)  # (K, N)

		part_kpts = kpts[:, self.part_ids]  # (K, N, 2): K people, N parts, 2D coordinates
		# select the corresponding parts
		y = part_kpts[:, :, 1].unsqueeze(1) * self.part_masks.to(self.device).unsqueeze(0)
		y = y * visible_kpt  # ignore the invisible parts

		y_min = y.clone()
		y_min[y_min == 0] = torch.inf
		y_min = y_min.amin(dim=-1)
		y_min[y_min == torch.inf] = 0
		y_min[:, 0] = bboxes[:, 1]

		y_max = y.amax(dim=-1)
		y_max = torch.concat([y_max, bboxes[:, 3:4]], dim=1)[:, 1:]

		# 
		patch_pixel_size = torch.div(bboxes[:, 3] - bboxes[:, 1], bbox_features.shape[-2])
		coor = torch.arange(bbox_features.shape[-2], device=bbox_features.device).unsqueeze(0).unsqueeze(0)  # (1, 1, H)
		coor = coor.expand(K, self.N, -1)

		y_min_stt = torch.div(y_min - bboxes[:, 1:2], patch_pixel_size[..., None])
		min_coor = torch.ge(coor, y_min_stt.long()[..., None])  # (K, N, H)

		y_max_stt = torch.div(y_max - bboxes[:, 1:2], patch_pixel_size[..., None])
		max_coor = torch.le(coor, y_max_stt.long()[..., None])  # (K, N, H)

		visible_part_map = (min_coor * max_coor).unsqueeze(-1).repeat(1, 1, 1, bbox_features.shape[-1]).long()
		visible_part_map = visible_part_map * visible_part.unsqueeze(-1).unsqueeze(-1)  # (K, N, H, W)
		
		global_features = F.adaptive_avg_pool2d(bbox_features, output_size=1).squeeze((-1, -2))

		masked_features = bbox_features.unsqueeze(1) * visible_part_map.unsqueeze(2)
		part_features = masked_features.sum(dim=(-2, -1)) / visible_part_map.sum(dim=(-2, -1))[..., None].clamp(min=1e-6)

		global_feature_norm = F.normalize(global_features, p=2, dim=-1)
		part_feature_norm = F.normalize(part_features, p=2, dim=-1)

		all_features = torch.concat([global_feature_norm.unsqueeze(1), part_feature_norm], dim=1)

		return all_features, visible_part
	
	def identify(self,
		all_features,
		visible_part,
	):
		scores = [
			classifier.predict(all_features[:, idx].cpu().detach().numpy())
			for idx, classifier in enumerate(self.classifiers)
		]
		avg_scores = np.mean(scores, axis=0)  # * visiblity_part_indicator
		target_id = np.argmax(avg_scores)

		return target_id

	def update(self,
		target_id,
		all_features,  # (K, N, ...)
		visible_part_indicator,  # (K, N, ...)
	):
		# TODO: self._update_resnet()
		
		K = all_features.shape[0]
		# update short-term memory	
		for k in range(K):
			for n in range(self.N+1):
				if n == 0:
					if visible_part_indicator[k, n-1] == 0:
						continue
				
				if k == target_id:  # positive
					if len(self.st_pos_memory[n]) > self.memory_size:
						self.st_pos_memory[n].pop(0)
					self.st_pos_memory[n].append(all_features[k, n])
				else:
					if len(self.st_neg_memory[n]) > self.memory_size:
						self.st_neg_memory[n].pop(0)
					self.st_neg_memory[n].append(all_features[k, n])
		
		# update long memory
		for k in range(K):
			for n in range(self.N+1):
				if n != 0:
					if visible_part_indicator[k, n-1] == 0:
						continue
				
				if k == target_id:  # positive
					if len(self.lt_pos_memory[n]) < self.memory_size:
						self.lt_pos_memory[n].append(all_features[k, n])
					else:
						rand_idx = np.random.randint(0, self.memory_size)
						self.lt_pos_memory[n][rand_idx] = all_features[k, n]
				else:
					if len(self.lt_pos_memory[n]) < self.memory_size:
						self.lt_pos_memory[n].append(all_features[k, n])
					else:
						rand_idx = np.random.randint(0, self.memory_size)
						self.lt_pos_memory[n][rand_idx] = all_features[k, n]
							
		# train Ridge classifier
		for idx, classifier in enumerate(self.classifiers):
			if len(self.st_pos_memory[idx]) < self.min_pos_example:
				continue
			
			X = torch.stack(self.st_pos_memory[idx]).cpu().detach().numpy()
			y = torch.ones(X.shape[0])
			
			if len(self.st_neg_memory[idx]) > 0:
				X_neg = torch.stack(self.st_neg_memory[idx]).cpu().detach().numpy()
				y_neg = -1 * torch.ones(X_neg.shape[0])
				X = np.concatenate([X, X_neg], axis=0)
				y = np.concatenate([y, y_neg], axis=0)
			
			classifier.fit(X, y)
