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
		self.memory_size = 120
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
		bboxes = torch.tensor(bboxes).to(self.device)	# (K, 4)
		kpts = torch.tensor(kpts).to(self.device)		# (K, 17, 2)
		confs = torch.tensor(confs).to(self.device)		# (K, 17)
		K = bboxes.shape[0]
		
		crop_imgs = [
			rgb_img[
				int(bbox[1]-bbox[3]/2):int(bbox[1]+bbox[3]/2), int(bbox[0]-bbox[2]/2):int(bbox[0]+bbox[2]/2)
			] for bbox in bboxes
		]
		bbox_features = self.resnet(
			torch.stack([
				self.transform(crop_img) for crop_img in crop_imgs
			]).to(self.device)
		)  # (K, D, H, W)

		# (K, 17) -> (K, N) -> (K, N, 9)  4 different parts from 9 joints
		selected_kpts = confs[:, self.part_ids]  # (K, N)
		# (K, 1, N) * (1, N, 9) -> (K, N, 9)
		visible_kpt = (selected_kpts.unsqueeze(1) * self.part_masks.to(self.device).unsqueeze(0))
		visible_kpt = visible_kpt > self.conf_thres
		visible_part = visible_kpt.amax(-1)  # (K, N)
		# print("visible_kpt:", visible_kpt.shape, visible_part.shape)

		# extract the y-coordinate of corresponding parts (each part has multiple keypoints)
		part_kpts = kpts[:, self.part_ids]  # (K, N, 2): K people, N parts, 2D coordinates
		# (K, 1, 9) * (1, N, 9)-> (K, N, 9)
		y = part_kpts[:, :, 1].unsqueeze(1) * self.part_masks.to(self.device).unsqueeze(0)
		y = y * visible_kpt  # ignore the invisible parts
		
		y_min = y.clone()
		y_min[y_min == 0] = torch.inf
		y_min = y_min.amin(dim=-1)
		y_min[y_min == torch.inf] = 0
		y_min[:, 0] = bboxes[:, 1] - bboxes[:, 3]/2

		# y_max = y.amin(dim=-1)
		y_max = y.amax(dim=-1)

		# range of a patch in the bbox
		patch_pixel_size = torch.div(bboxes[:, 3], bbox_features.shape[-2])
		coor = torch.arange(bbox_features.shape[-2], device=bbox_features.device).unsqueeze(0).unsqueeze(0)  # (1, 1, H)
		coor = coor.expand(K, self.N, -1)

		y_min_stt = torch.div(y_min - (bboxes[:, 1:2] - bboxes[:, 3:4]/2), patch_pixel_size[..., None])
		min_coor = torch.ge(coor, y_min_stt.long()[..., None])  # (K, N, H)

		y_max_stt = torch.div(y_max - (bboxes[:, 1:2] - bboxes[:, 3:4]/2), patch_pixel_size[..., None])
		max_coor = torch.le(coor, y_max_stt.long()[..., None])  # (K, N, H)

		visible_part_map = (min_coor * max_coor).unsqueeze(-1).repeat(1, 1, 1, bbox_features.shape[-1]).long()
		visible_part_map = visible_part_map * visible_part.unsqueeze(-1).unsqueeze(-1)  # (K, N, H, W)

		masked_features = bbox_features.unsqueeze(1) * visible_part_map.unsqueeze(2)
		part_features = masked_features.sum(dim=(-2, -1)) / visible_part_map.sum(dim=(-2, -1))[..., None].clamp(min=1e-6)
		
		global_features = F.adaptive_avg_pool2d(bbox_features, output_size=1).squeeze((-1, -2))

		all_features = torch.concat([global_features.unsqueeze(1), part_features], dim=1)

		return all_features, visible_part
	
	def identify(self,
		all_features,  # (K, N+1, D)
		visible_part,  # (K, N)
		threshold=0.5
	):
		global_visible_part = visible_part.amax(axis=-1, keepdim=True)
		_visible_part = torch.cat([global_visible_part, visible_part], dim=-1)

		K = _visible_part.shape[0]

		scores = []
		for idx, classifier in enumerate(self.classifiers):
			if len(self.st_pos_memory[idx]) >= self.min_pos_example:
				# print(all_features[:, idx].shape, _visible_part[:, idx].shape)
				scores.append(classifier.predict(all_features[:, idx]) * _visible_part[:, idx].numpy())
			else:
				scores.append(np.zeros(K))
		
		avg_scores = np.divide(
			np.sum(scores, axis=0),
			_visible_part.sum(-1).numpy()
		)
		target_id = np.argmax(avg_scores)
		# print(np.array(scores), avg_scores[target_id])
		if avg_scores[target_id] < threshold: return None

		return target_id

	def update(self,
		target_id,
		all_features,  # (K, N, ...)
		visible_part,  # (K, N, ...)
	):	
		K = all_features.shape[0]

		# update classifiers using short-term memory
		for k in range(K):
			for n in range(self.N+1):
				if n == 0 and visible_part[k].any() == 0: continue
				if n > 0 and visible_part[k, n-1] == 0: continue
				
				if k == target_id:  # positive
					# print("k", k, n)
					if len(self.st_pos_memory[n]) > self.memory_size:
						self.st_pos_memory[n].pop(0)
					self.st_pos_memory[n].append(all_features[k, n])
				else:
					if len(self.st_neg_memory[n]) > self.memory_size:
						self.st_neg_memory[n].pop(0)
					self.st_neg_memory[n].append(all_features[k, n])
		
		for idx, classifier in enumerate(self.classifiers):
			if len(self.st_pos_memory[idx]) < self.min_pos_example:
				# print("skip", idx)
				continue
			
			X = torch.stack(self.st_pos_memory[idx]).cpu().detach().numpy()
			y = torch.ones(X.shape[0])
			
			if len(self.st_neg_memory[idx]) > 0:
				X_neg = torch.stack(self.st_neg_memory[idx]).cpu().detach().numpy()
				y_neg = -1 * torch.ones(X_neg.shape[0])
				X = np.concatenate([X, X_neg], axis=0)
				y = np.concatenate([y, y_neg], axis=0)
			
			# print('train', idx, X.shape, y.shape)

			classifier.fit(X, y)
		
		# # update ResNet using long-term memory
		# for k in range(K):
		# 	for n in range(self.N+1):
		# 		# if n != 0 or visible_part[k, n-1] == 0:
		# 		# 	continue
				
		# 		if k == target_id:  # positive
		# 			if len(self.lt_pos_memory[n]) < self.memory_size:
		# 				self.lt_pos_memory[n].append(all_features[k, n])
		# 			else:
		# 				rand_idx = np.random.randint(0, self.memory_size)
		# 				self.lt_pos_memory[n][rand_idx] = all_features[k, n]
		# 		else:
		# 			if len(self.lt_pos_memory[n]) < self.memory_size:
		# 				self.lt_pos_memory[n].append(all_features[k, n])
		# 			else:
		# 				rand_idx = np.random.randint(0, self.memory_size)
		# 				self.lt_pos_memory[n][rand_idx] = all_features[k, n]
