import numpy as np 
from .kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque


class Tracks:
	
	def __init__(self, detection, trackId):
		super(Tracks, self).__init__()
		self.KF = KalmanFilter()
		self.KF.predict()
		self.KF.correct(np.matrix(detection).reshape(2, 1))
		self.trace = deque(maxlen=20)
		self.prediction = detection.reshape(1, 2)
		self.trackId = trackId
		self.skipped_frames = 0

	def predict(self, detection):
		self.prediction = np.array(self.KF.predict()).reshape(1, 2)
		self.KF.correct(np.matrix(detection).reshape(2, 1))


class Tracker:
	
	def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.trackId = 0
		self.tracks = []

	def get_track_position(self, track_id):
		for track in self.tracks:
			if track.trackId == track_id:
				return [
					track.trace[-1][0, 0],
					track.trace[-1][0, 1]
				]
		else:
			return None

	def update(self, detections):
		# init if there is no track
		# each detected person is assigned with a unique trackID
		if len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				track = Tracks(detections[i], self.trackId)
				self.trackId += 1
				self.tracks.append(track)

		# associate position of current and past based on spatial information
		N = len(self.tracks)
		M = len(detections)
		cost = []  # (len(self.tracks), len(detections))
		for i in range(N):
			diff = np.linalg.norm(
				self.tracks[i].prediction - detections.reshape(-1, 2),
				axis=1
			)
			cost.append(diff)

		cost = np.array(cost)*0.1
		row, col = linear_sum_assignment(cost)
		assignment = [-1]*N

		for r, c in zip(row, col):
			if cost[r, c] > self.dist_threshold: continue
			assignment[r] = c

		# delete tracks with too many skipped frames
		del_tracks = [
			i
			for i, track in enumerate(self.tracks)
			if track.skipped_frames > self.max_frame_skipped
		]

		for i in sorted(del_tracks, reverse=True):
			del self.tracks[i]
			del assignment[i]

		# add new track
		for i in range(M):
			if i not in assignment:
				track = Tracks(detections[i], self.trackId)
				self.trackId += 1
				self.tracks.append(track)

		# perform Kalman filter
		for i in range(len(assignment)):
			if(assignment[i] != -1):
				self.tracks[i].skipped_frames = 0
				self.tracks[i].predict(detections[assignment[i]])
			else:
				self.tracks[i].skipped_frames += 1
			self.tracks[i].trace.append(self.tracks[i].prediction)
