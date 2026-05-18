import warnings
from typing import Callable, Dict, List, Optional, Tuple
from functools import cached_property
import numpy as np
import math
import torch
from torchvision.ops import masks_to_boxes
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from sam2.modeling.sam2_base import NO_OBJ_SCORE
from sam2.sam2_video_predictor import SAM2VideoPredictor


def mask_centroid(mask):  
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    cx = xs.float().mean()
    cy = ys.float().mean()
    return torch.stack([cx, cy]).cpu().numpy()

def batch_mask_centroids(masks):
    N, H, W = masks.shape
    m = masks.float()

    ys = torch.arange(H, device=m.device, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, device=m.device, dtype=torch.float32).view(1, 1, W)

    counts = m.sum(dim=(1, 2))                        
    cx = (m * xs).sum(dim=(1, 2)) / counts.clamp(min=1)
    cy = (m * ys).sum(dim=(1, 2)) / counts.clamp(min=1)

    centroids = torch.stack([cx, cy], dim=1) 
    return centroids
    
def _bbox_center(box: np.ndarray) -> np.ndarray:
    """Compute center of box."""
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


def _center_distance(center_a: np.ndarray, center_b: np.ndarray) -> float:
    """Euclidean distance between two center points."""
    return math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


def _iou_bbox(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _remove_contained_boxes(
    boxes: np.ndarray,
    confs: np.ndarray,
    containment_threshold: float = 0.8,
) -> np.ndarray:
    """
    Remove boxes that are mostly contained inside another box.
    """
    if len(boxes) <= 1:
        return boxes

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue

            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)

            smaller_area = min(areas[i], areas[j])
            if smaller_area == 0:
                continue

            if inter / smaller_area > containment_threshold:
                if confs[i] < confs[j]:
                    keep[i] = False
                else:
                    keep[j] = False

    return boxes[np.array(keep)]


class SAM2YOLOVideoPredictor(SAM2VideoPredictor):
    """
    SAM2VideoPredictor subclass that integrates an external detector
    into the propagation loop for robust cell tracking.

    
    """

    # start temporary object ID
    _TEMP_ID_START = 1000000

    def __init__(
        self,
        # --- Matching ---
        match_iou_threshold: float = 0.4,
        # --- Mask merging (cells joining in reverse) ---
        mask_containment_threshold: float = 0.8,
        # --- Removed-object memory ---
        removed_max_age: int = 16,
        # --- Temporary objects ---
        temp_object_maturity: int = 4,
        swap_distance_threshold: float = 100.0,
        max_age_without_yolo_box: int = 6,
        correction_iou_threshold: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.match_iou_threshold = match_iou_threshold
        self.mask_containment_threshold = mask_containment_threshold
        self.removed_max_age = removed_max_age
        self.temp_object_maturity = temp_object_maturity
        self.swap_distance_threshold = swap_distance_threshold
        self.max_age_without_yolo_box =max_age_without_yolo_box
        self.correction_iou_threshold=correction_iou_threshold
        self._model_dtype_cache = None

    @staticmethod
    def _init_tracking_structures(inference_state: dict):
        """Initialise the extra bookkeeping dicts on the inference state."""
        inference_state["_removed_objects"] = {}   # obj_id -> {frame_idx, centroid, age}
        inference_state["_next_temp_id"] = SAM2YOLOVideoPredictor._TEMP_ID_START
        inference_state["_next_id"] = 1
        inference_state["_temp_objects"] = {} 
        inference_state["_objs_with_no_match"] = {}
        inference_state["_swapped_objs"] = {}
        inference_state["_merge_event"] = {}
        
    @staticmethod
    def is_temporary_object(inference_state: dict, obj_id: int) -> bool:
        """Return True if obj_id is a temporary (unconfirmed) object."""
        return obj_id in inference_state.get("_temp_objects", {})

    def _swap_obj_id(
        self,
        inference_state: dict,
        old_id: int,
        new_id: int,
    ):
        obj_idx = self._obj_id_to_idx(inference_state, old_id)

        pos = inference_state["obj_ids"].index(old_id)
        inference_state["obj_ids"][pos] = new_id
    
        inference_state["obj_id_to_idx"].pop(old_id)
        inference_state["obj_id_to_idx"][new_id] = obj_idx
        
        inference_state["obj_idx_to_id"][obj_idx] = new_id
        inference_state["obj_id_to_idx"] = OrderedDict(
            sorted(inference_state["obj_id_to_idx"].items(), key=lambda kv: kv[1])
        )
        inference_state["obj_idx_to_id"] = OrderedDict(
            sorted(inference_state["obj_idx_to_id"].items(), key=lambda kv: kv[0])
        )
        
        inference_state["_swapped_objs"][new_id] = old_id


    def _register_removed_object(
        self,
        inference_state: dict,
        obj_id: int,
        frame_idx: int,
        centroid: Optional[np.ndarray],
    ):
        """
        Record an object that was just  merged so it can
        potentially be reconnected later via a temporary object.
        """
        if self.is_temporary_object(inference_state, obj_id):
            inference_state["_temp_objects"].pop(obj_id, None)
            return

        inference_state["_removed_objects"][obj_id] = {
            "frame_idx": frame_idx,
            "centroid": centroid.copy() if centroid is not None else None,
            "age": 0,
        }

    def _age_removed_objects(self, inference_state: dict):
        """Increment age of every remembered removed object; purge stale ones."""
        removed = inference_state.get("_removed_objects", {})
        to_delete = []
        for obj_id, info in removed.items():
            info["age"] += 1
            if info["age"] > self.removed_max_age:
                to_delete.append(obj_id)
        for obj_id in to_delete:
            del removed[obj_id]


    def _create_temporary_object(
        self,
        inference_state: dict,
        frame_idx: int,
        box: np.ndarray,
    ) -> int:
        """Spawn a new temporary SAM2 object from a YOLO box."""
        temp_id = inference_state["_next_temp_id"]
        inference_state["_next_temp_id"] += 1
        inference_state["_temp_objects"][temp_id] = {
            "frame_idx": frame_idx,
            "centroid": _bbox_center(box),
            "age": 0,
        }
        # print("new tempral object created" , temp_id, "with box", len(box), flush=True)
        self.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=temp_id,
            box=box,
            clear_old_points=True,
            normalize_coords=True,
        )
        return temp_id

    def _age_temporary_objects(self, inference_state: dict):
        """Increment the age counter for every living temporary object."""
        for temp_id, info in inference_state.get("_temp_objects", {}).items():
            info["age"] += 1

    def _evaluate_temporary_objects(
        self,
        inference_state: dict,
        frame_idx: int,
    ):
        """
        For every temporary object that has reached maturity, try to reconnect it to the
        closest removed object within distance.
        """
        temp_objects = inference_state.get("_temp_objects", {})
        
        removed = inference_state.get("_removed_objects", {})
        if not removed:
            return

        to_promote: List[Tuple[int, int]] = []  # (temp_id, original_id)
        claimed_orig: set = set()
        
        for temp_id, info in list(temp_objects.items()):
            if info["age"] < self.temp_object_maturity:
                continue
                
            temp_centroid = info["centroid"]

            best_orig_id = None
            best_dist = self.swap_distance_threshold  # acts as upper bound

            for orig_id, info_removed in removed.items():
                if orig_id in claimed_orig:
                    continue
                
                delta_age = info_removed['age'] - info["age"]
                if info_removed["centroid"] is None or info_removed['age'] == 0 or delta_age == 0 or delta_age <= (-3):
                    continue
                dist = _center_distance(temp_centroid, info_removed["centroid"])
                if dist < best_dist:
                    best_dist = dist
                    best_orig_id = orig_id

            if best_orig_id is not None:
                to_promote.append((temp_id, best_orig_id))
                claimed_orig.add(best_orig_id)
        # print("To promote ",  to_promote, flush=True)
        
        for temp_id, orig_id in to_promote:
            self._swap_obj_id(inference_state, old_id=temp_id, new_id=orig_id)
            # print("swap temp->", temp_id, " new_id ->" , orig_id, flush=True)
            temp_objects.pop(temp_id, None)
            if temp_id in inference_state["_objs_with_no_match"]:
                inference_state["_objs_with_no_match"].pop(temp_id)


            removed.pop(orig_id, None)
            


    @staticmethod
    def load_frames_from_folder(frames_dir: str) -> List[np.ndarray]:
        """Load all JPEG/PNG frames from a folder, sorted by filename."""
        import os
        from PIL import Image

        extensions = {".jpg", ".jpeg", ".png"}
        frame_files = sorted(
            f
            for f in os.listdir(frames_dir)
            if os.path.splitext(f)[1].lower() in extensions
        )
        if len(frame_files) == 0:
            raise ValueError(f"No image files found in {frames_dir}")

        frames = []
        for fname in frame_files:
            img = Image.open(os.path.join(frames_dir, fname)).convert("RGB")
            frames.append(np.array(img))
        return frames

    @torch.inference_mode()
    def init_state_with_detector(
        self,
        frames_dir: str,
        detector_fn: Callable,
        video_frames: Optional[List[np.ndarray]] = None,
        init_frame_idx: int = -1,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Tuple[dict, List[np.ndarray]]:
        
        inference_state = self.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )

        self._init_tracking_structures(inference_state)

        if video_frames is None:
            video_frames = self.load_frames_from_folder(frames_dir)

        num_frames = inference_state["num_frames"]
        if init_frame_idx < 0:
            init_frame_idx = num_frames + init_frame_idx

        yolo_boxes = detector_fn(video_frames[init_frame_idx])
        if yolo_boxes is None or len(yolo_boxes) == 0:
            warnings.warn(
                f"Detector found no objects on frame {init_frame_idx}. "
                "Tracking will start with zero objects."
            )
            return inference_state, video_frames

        for det_idx, box in enumerate(yolo_boxes):
            obj_id = inference_state["_next_id"]
            inference_state["_next_id"] += 1
            
            self.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=init_frame_idx,
                obj_id=obj_id,
                box=box,
                clear_old_points=True,
                normalize_coords=True,
            )

        self._encode_temp_outputs(inference_state)

        return inference_state, video_frames


    def _encode_temp_outputs(self, inference_state: dict):
        """
        Encode all pending temporary outputs into memory.
        """
        model_dtype = self._model_dtype
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

            for is_cond in [False, True]:
                storage_key = (
                    "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                )
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    if out["maskmem_features"] is None:
                        high_res_masks = torch.nn.functional.interpolate(
                            out["pred_masks"].to(inference_state["device"]).float(),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,
                            high_res_masks=high_res_masks,
                            object_score_logits=out["object_score_logits"],
                            is_mask_from_pts=True,
                        )
                        if maskmem_features is not None:
                            maskmem_features = maskmem_features.to(model_dtype)
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc
                    elif out["maskmem_features"] is not None:
                        out["maskmem_features"] = out["maskmem_features"].to(model_dtype)

                    obj_output_dict[storage_key][frame_idx] = out

                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )

                obj_temp_output_dict[storage_key].clear()

            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)


    def _get_obj_score_logit(
        self,
        inference_state: dict,
        obj_id: int,
        frame_idx: int,
    ) -> Optional[float]:
        """
        Return the object-score logit for cell obj_id at frame_dx.

        This is the raw logit from object_score_logits stored in the
        per-object output dict.

        Returns None if no output exists for this obj_id.
        """
        out = self._get_frame_output(inference_state, obj_id, frame_idx)
        if out is None:
            return None
        logits = out.get("object_score_logits")
        return float(logits.flatten()[0]) if logits is not None else None

    def _get_video_res_masks(
        self,
        inference_state: dict,
        frame_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Collect pred_masks for all obj_ids at frame_idx,
        interpolate to video resolution, and return the combined tensor.
        Returns None if no masks found.
        """
        obj_ids = list(inference_state["obj_ids"])
        
        pred_masks_per_obj = []
        for obj_id in obj_ids:
            out = self._get_frame_output(inference_state, obj_id, frame_idx)
            if out is not None:
                pred_masks_per_obj.append(
                    out["pred_masks"].to(inference_state["device"], non_blocking=True)
                )
    
        if not pred_masks_per_obj:
            return None
    
        all_pred_masks = (
            torch.cat(pred_masks_per_obj, dim=0)
            if len(pred_masks_per_obj) > 1
            else pred_masks_per_obj[0]
        )
        _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)

                
        return video_res_masks

    def _get_frame_output(
        self,
        inference_state: dict,
        obj_id: int,
        frame_idx: int,
    ) -> Optional[dict]:
        """
        Return the stored output dict for obj_id at frame_idx
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
        if out is None:
            out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        return out

    def find_containment_pairs(
        self,
        video_res_masks: torch.Tensor,
        containment_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        
        n_obj = video_res_masks.shape[0]
        if n_obj < 2:
            return [], torch.zeros(n_obj, device=video_res_masks.device)
    
        masks_bool = (video_res_masks > 0)[:, 0]              # (N, H, W) bool
        flat = masks_bool.view(n_obj, -1).float()             # (N, H*W)
        areas = flat.sum(dim=1)                                # (N,)
        inter = flat @ flat.T                                  # (N, N)
        del flat
        
        i_idx, j_idx = torch.triu_indices(
            n_obj, n_obj, offset=1, device=inter.device
        )
        pair_inter = inter[i_idx, j_idx]
        pair_min = torch.minimum(areas[i_idx], areas[j_idx])
        del inter
        
        keep = (pair_min > 0) & (pair_inter > containment_threshold * pair_min)
        pairs = torch.stack([i_idx[keep], j_idx[keep]], dim=1).cpu().tolist()
        
        return [(int(i), int(j)) for i, j in pairs], areas
        
    def _apply_corrections(
        self,
        inference_state: dict,
        frame_idx: int,
        video_res_masks: torch.Tensor,
        yolo_boxes: np.ndarray,
    ) -> bool:
        """
        Correction steps for cells with YOLO and SAM2 reprompting, object removal, adding temporary objects
        """
        # print("frame idx", frame_idx, flush=True)
        if yolo_boxes is None or len(yolo_boxes) == 0:
            return

        obj_ids = list(inference_state["obj_ids"])
        n_obj = len(obj_ids)
        if n_obj == 0:
            return

        # Step 1: greedy match cells with boxes
        # reprompt with negative poits and correct box 
        # remove cell with no yolo box for n frames
        masks = (video_res_masks > 0)[:, 0]            
        non_empty = masks.flatten(1).any(dim=1)        
        
        sam_bboxes_current: Dict[int, np.ndarray] = {}
        if non_empty.any():
            boxes = masks_to_boxes(masks[non_empty])   
            boxes_np = boxes.cpu().numpy()             
            kept_idx = non_empty.nonzero(as_tuple=True)[0].tolist()
            for i, pos in enumerate(kept_idx):
                sam_bboxes_current[obj_ids[pos]] = boxes_np[i]

        # Previous-frame centroids (frame_idx+1 because we propagate in reverse)
        prev_frame_idx = frame_idx + 1
        sam_centroids_prev: Dict[int, np.ndarray] = {}
        if prev_frame_idx < inference_state["num_frames"]:
            prev_masks = self._get_video_res_masks(inference_state, prev_frame_idx)
            if prev_masks is not None:
                prev_obj_ids = list(inference_state["obj_ids"])  # snapshot at prev frame would be more correct
                prev_binary = prev_masks[:, 0] > 0
                prev_centroids = batch_mask_centroids(prev_binary).cpu().numpy()
                for pos, oid in enumerate(prev_obj_ids):
                    sam_centroids_prev[oid] = prev_centroids[pos]
                del prev_masks, prev_binary
        yolo_centers = np.array([_bbox_center(ybox) for ybox in yolo_boxes])

        # print("current boxes",sam_bboxes_current.keys(), flush=True)
        # print("prev boxes", sam_centroids_prev.keys(), flush=True)
        
        # for each object, collect candidate YOLO indices
        obj_candidates: Dict[int, set] = {}
        for obj_id, curr_bbox in sam_bboxes_current.items():
            candidates: set = set()
            for j, ybox in enumerate(yolo_boxes):
                iou_score = _iou_bbox(curr_bbox, ybox)
                if iou_score >= self.match_iou_threshold:
                    candidates.add(j)
            if candidates:
                obj_candidates[obj_id] = candidates
        # print("candidates ",obj_candidates, flush=True)
        
        # Greedy assignment, always assign best candidate
        all_matches: Dict[int, int] = {}   # obj_id -> yolo_idx
        unmatched_objs = set(obj_candidates.keys())

        while unmatched_objs:
            best_obj = None
            best_nearest_dist = float('inf')

            for obj_id in unmatched_objs:
                candidates = obj_candidates[obj_id]
                if not candidates:
                    continue
                prev_centroid = sam_centroids_prev.get(obj_id)
                if prev_centroid is not None:
                    nearest_dist = min(
                        _center_distance(prev_centroid, yolo_centers[j])
                        for j in candidates
                    )
                else:
                    curr_center = _bbox_center(sam_bboxes_current[obj_id])
                    nearest_dist = min(
                        _center_distance(curr_center, yolo_centers[j])
                        for j in candidates
                    )

                if nearest_dist < best_nearest_dist:
                    best_nearest_dist = nearest_dist
                    best_obj = obj_id

            if best_obj is None:
                break

            candidates = obj_candidates[best_obj]
            prev_centroid = sam_centroids_prev.get(best_obj)
            if prev_centroid is not None:
                best_j = min(
                    candidates,
                    key=lambda j: _center_distance(prev_centroid, yolo_centers[j]),
                )
            else:
                best_j = max(
                    candidates,
                    key=lambda j: _iou_bbox(
                        sam_bboxes_current[best_obj], yolo_boxes[j]
                    ),
                )

            all_matches[best_obj] = best_j
            unmatched_objs.remove(best_obj)

            # Remove this YOLO box from all other candidates
            for obj_id in unmatched_objs:
                if len(obj_candidates[obj_id]) == 1 and best_j in obj_candidates[obj_id]:
                    # print("obj_candidates which best_j wont be discarded ", obj_id," best j ", best_j, flush=True)
                    continue
                obj_candidates[obj_id].discard(best_j)
        # print("1 all matches", all_matches, flush=True)

        # Check objects with no yolo box 

        no_match_remove_ids = []
        _objs_with_no_match = inference_state["_objs_with_no_match"]
        # print("_objs_with_no_match ", _objs_with_no_match, flush=True)
        for obj_id in obj_ids:
            if obj_id in all_matches and obj_id in _objs_with_no_match :
                _objs_with_no_match.pop(obj_id)
            elif obj_id not in all_matches and obj_id in inference_state["_objs_with_no_match"]:
                count = _objs_with_no_match[obj_id]['no_yolo_box_count']
                count = count + 1
                if count > self.max_age_without_yolo_box:
                    no_match_remove_ids.append(obj_id) 
                _objs_with_no_match[obj_id]['no_yolo_box_count'] = count
            elif obj_id not in all_matches and obj_id not in inference_state["_objs_with_no_match"]:
                _objs_with_no_match[obj_id] = {}
                _objs_with_no_match[obj_id]["no_yolo_box_count"] = 1
                _objs_with_no_match[obj_id]["centroid"] = sam_centroids_prev.get(obj_id)

        # remove objects without yolo box for n consencutive frames            
        for obj_id in no_match_remove_ids:
            # print("Removing obj (no YOLO match for 6 frames)", obj_id, flush=True)
            centroid = _objs_with_no_match[obj_id]["centroid"]
            self._register_removed_object(inference_state, obj_id, frame_idx, centroid)
            self.remove_object(
                inference_state, obj_id, strict=False, need_output=False
            )
            # print("removed object", obj_id, flush=True)
            _objs_with_no_match.pop(obj_id)
        
        # reprompt matched objects 

        for obj_id, yolo_idx in all_matches.items():
            sam_bbox = sam_bboxes_current.get(obj_id)
            box = yolo_boxes[yolo_idx]    
            
            cands = obj_candidates.get(obj_id, set())
            other_candidates = [j for j in cands if j != yolo_idx]
            if not other_candidates and _iou_bbox(sam_bbox, box) > self.correction_iou_threshold:
                continue
            
            points = None
            labels = None
            if other_candidates:
                points = np.array(
                    [_bbox_center(yolo_boxes[j]) for j in other_candidates],
                    dtype=np.float32,
                )
                labels = np.zeros(len(other_candidates), dtype=np.int32)
            
            self.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
                points=points,
                labels=labels,
                clear_old_points=True,
                normalize_coords=True,
            )

        
        # Step 2: create temporary objects from unused yolo boxes and re-assgin mature temporary objects
        # print("matched_yolo_indices", matched_yolo_indices, flush=True)
        # print("yolo_boxes", yolo_boxes, flush=True)
        matched_yolo_indices = set(all_matches.values())  
        for j, box in enumerate(yolo_boxes):
            if j not in matched_yolo_indices:
                self._create_temporary_object(inference_state, frame_idx, box)       
                # print("object creation ", j,tmp_id, flush=True)
                
        self._encode_temp_outputs(inference_state)
        self._evaluate_temporary_objects(inference_state, frame_idx)
        
        video_res_masks = self._get_video_res_masks(inference_state, frame_idx)

        
        # Step 3: merge contained masks and perform cell removal
        obj_ids = list(inference_state["obj_ids"])
        n_obj = len(obj_ids)
        
        candidate_pairs, _areas = self.find_containment_pairs(
            video_res_masks, self.mask_containment_threshold
        )
        
        daughter_set: set = set()
        parent_set: set = set()
        to_remove_ids = []
        
        for pos_i, pos_j in candidate_pairs:
            oid_i = obj_ids[pos_i]
            oid_j = obj_ids[pos_j]
        
            if oid_i in daughter_set or oid_j in daughter_set:
                continue
        
            j_logits = self._get_obj_score_logit(inference_state, oid_j, frame_idx)
            i_logits = self._get_obj_score_logit(inference_state, oid_i, frame_idx)
        
            if oid_j in parent_set and oid_i in parent_set:
                continue
            if oid_j in parent_set:
                oid_to_be_removed = oid_i
                oid_kept = oid_j
                pos_to_be_removed = pos_i
            elif oid_i in parent_set:
                oid_to_be_removed = oid_j
                oid_kept = oid_i
                pos_to_be_removed = pos_j
            elif self.is_temporary_object(inference_state, oid_j) and not self.is_temporary_object(inference_state, oid_i):
                oid_to_be_removed = oid_j
                oid_kept = oid_i
                pos_to_be_removed = pos_j
            elif self.is_temporary_object(inference_state, oid_i) and not self.is_temporary_object(inference_state, oid_j):
                oid_to_be_removed = oid_i
                oid_kept = oid_j
                pos_to_be_removed = pos_i
            elif i_logits > j_logits:
                oid_to_be_removed = oid_j
                oid_kept = oid_i
                pos_to_be_removed = pos_j
            else:
                oid_to_be_removed = oid_i
                oid_kept = oid_j
                pos_to_be_removed = pos_i
        
            daughter_set.add(oid_to_be_removed)
            parent_set.add(oid_kept)
        
            mask_gpu = video_res_masks[pos_to_be_removed, 0] > 0
            centroid_removed = mask_centroid(mask_gpu)
            to_remove_ids.append([oid_to_be_removed, oid_kept])
            self._register_removed_object(
                inference_state, oid_to_be_removed, frame_idx, centroid_removed
            )
                            
                       
        # print("to be removed ", to_remove_ids, flush=True)
        # print("1 obj_id_to_idx ", inference_state["obj_id_to_idx"], flush=True)
        # print("1 obj_idx_to_id ", inference_state["obj_idx_to_id"], flush=True)
        
        for obj_id_to_remove, obj_kept in to_remove_ids:
                self.remove_object(
                    inference_state, obj_id_to_remove, strict=False, need_output=False
                )

                if frame_idx not in inference_state["_merge_event"]:
                    inference_state["_merge_event"][frame_idx] = []
                # print(obj_kept)
                inference_state["_merge_event"][frame_idx].append({
                    "parent": obj_kept,
                    "daughter": obj_id_to_remove
                })
                if obj_id_to_remove in inference_state["_objs_with_no_match"]:
                    inference_state["_objs_with_no_match"].pop(obj_id_to_remove)
        
        # print("2 obj_id_to_idx ", inference_state["obj_id_to_idx"], flush=True)
        # print("2 obj_idx_to_id ", inference_state["obj_idx_to_id"], flush=True)
        # print("removed objects" , inference_state["_removed_objects"], flush=True)
       
        return video_res_masks

    @cached_property
    def _model_dtype(self):
        for module in (self.memory_attention, self.sam_mask_decoder):
            try:
                return next(module.parameters()).dtype
            except StopIteration:
                continue
        return torch.float32

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Override parent to use model dtype instead of hardcoded bfloat16.
        
        """
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        storage_device = inference_state["storage_device"]
        model_dtype = self._model_dtype
        maskmem_features = maskmem_features.to(model_dtype)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """
        Override parent function to fix dtype mismatch.
        
        """
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        storage_device = inference_state["storage_device"]
        model_dtype = self._model_dtype

        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(model_dtype)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)

        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            from sam2.utils.misc import fill_holes_in_mask_scores

            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)

        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]

        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    # ------------------------------------------------------------------
    # Main propagation loop
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        detector_fn: Optional[Callable] = None,
        video_frames: Optional[List[np.ndarray]] = None,
        reverse: bool = True,
        start_frame_idx=-1,
        max_frame_num_to_track=None,
    ):
        """
        Main function to lead YOLO, all SAM2 components and corrections.

        It first initializes tracking structures, then run per frame inference with SAM2.
        At each frame it runs detector and corrections.
        
        Yields:
            (frame_idx, obj_ids, video_res_masks, yolo_boxes) per frame.
        """
        num_frames = inference_state["num_frames"]
        if start_frame_idx < 0:
            start_frame_idx = num_frames + start_frame_idx
        
        if detector_fn is None:
            raise RuntimeError("YOLO detector not initialized")

        processing_order = range(start_frame_idx, -1, -1)

        for frame_idx in tqdm(
            processing_order, desc="propagate reverse (YOLO-assisted)"
        ):
            self._age_removed_objects(inference_state)
            self._age_temporary_objects(inference_state)

            obj_ids = list(inference_state["obj_ids"])
            batch_size = len(obj_ids)

            # step 1: SAM2 tracking
            pred_masks_per_obj: List[Optional[torch.Tensor]] = [None] * batch_size
            obj_scores = {}
            for pos, obj_id in enumerate(obj_ids):
                obj_idx = self._obj_id_to_idx(inference_state, obj_id)
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    
                    current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                    pred_masks = current_out["pred_masks"].to(
                        inference_state["device"], non_blocking=True
                    )
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=True,
                        run_mem_encoder=True,
                    )
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": True
                }
                pred_masks_per_obj[pos] = pred_masks


            video_res_masks = self._get_video_res_masks(inference_state, frame_idx)

            # step 2: run YOLO 
            frame_yolo_boxes = detector_fn(video_frames[frame_idx])
            if frame_yolo_boxes is None:
                frame_yolo_boxes = np.empty((0, 4), dtype=np.float32)

            # step 3: apply corrections
            self._apply_corrections(
                inference_state=inference_state,
                frame_idx=frame_idx,
                video_res_masks=video_res_masks,
                yolo_boxes=frame_yolo_boxes,
            )

            video_res_masks = self._get_video_res_masks(inference_state, frame_idx)
            
            # step 4: yield results 
            yield frame_idx, list(inference_state["obj_ids"]), video_res_masks, frame_yolo_boxes


# ------------------------------------------------------------------
# YOLO detector wrapper
# ------------------------------------------------------------------


def make_yolo_detector(
    model_path: str = "yolo11l.pt",
    confidence: float = 0.4,
    classes: Optional[List[int]] = None,
    device: str = "cuda",
    containment_threshold: float = 0.8,
) -> Callable:
    """
    Create a detector_fn compatible with SAM2YOLOVideoPredictor.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)

    def detect(frame_rgb: np.ndarray) -> np.ndarray:
        results = model(
            frame_rgb, conf=confidence, classes=classes, device=device, verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        if len(boxes) > 1 and containment_threshold < 1.0:
            boxes = _remove_contained_boxes(boxes, confs, containment_threshold)
        return boxes

    return detect