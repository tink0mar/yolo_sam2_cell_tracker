import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from collections import defaultdict
import shutil
import argparse
import re

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_yolo_video_cell_tracker import SAM2YOLOVideoPredictor, make_yolo_detector


if torch.cuda.is_available():
    device = torch.device("cuda")

temp_object_start_id = 1000000

def has_jpg_files(folder_path):
    if not os.path.exists(folder_path):
        return False
    jpg_extensions = ('.jpg', '.jpeg')
    return any(f.lower().endswith(jpg_extensions) for f in os.listdir(folder_path))


def remove_tmp_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Removed: {folder_path}")
    else:
        print(f"No tmp folder to remove at {tmp_path}")


def convert_tif_to_jpg(folder_path, tmp_name='tmp'):
    tif_extensions = ('.tif', '.tiff')

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None

    parts = Path(folder_path).parts
    dataset = parts[-2]    
    sequence = parts[-1]   
    
    tmp_path = os.path.join("./", f"{dataset}-{sequence}-{tmp_name}")
    # Skip if JPGs already exist in tmp
    if has_jpg_files(tmp_path):
        print(f"JPGs already exist in '{tmp_path}', skipping conversion.")
        return tmp_path

    files = os.listdir(folder_path)
    tif_files = [f for f in files if f.lower().endswith(tif_extensions)]
    if not tif_files:
        print(f"No TIF files found in '{folder_path}'.")
        return None

    os.makedirs(tmp_path, exist_ok=True)

    for filename in tif_files:
        tif_path = os.path.join(folder_path, filename)
        image = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to read {filename}, skipping.")
            continue

        if image.dtype != 'uint8':
            image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(filename)[0]
        if base_name.startswith('t'):
            base_name = base_name[1:]

        jpg_path = os.path.join(tmp_path, base_name + '.jpg')
        cv2.imwrite(jpg_path, image_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"Converted TIFs in '{folder_path}' to JPGs in '{tmp_path}'")
    return tmp_path

def validate_res_track(mask_dir, prefix="mask"):
    tracks = []
    with open(f"{mask_dir}/res_track.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track_id, start, end, parent = map(int, line.split())
            tracks.append((track_id, start, end, parent))

    min_frame = min(t[1] for t in tracks)
    max_frame = max(t[2] for t in tracks)

    n_files = sum(1 for _ in Path(mask_dir).glob(f"*tif"))
    digits = 4 if n_files > 999 else 3

    print(f"loading masks for validation {mask_dir}")
    frame_ids = {} 
    for frame_idx in range(min_frame, max_frame + 1):
        mask_path = os.path.join(mask_dir, f"{prefix}{frame_idx:0{digits}d}.tif")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"  warning: could not load {mask_path}")
            frame_ids[frame_idx] = set()
            continue
        frame_ids[frame_idx] = set(np.unique(mask).tolist())

    issues = defaultdict(list)
    for track_id, start, end, parent in tracks:
        for frame_idx in range(start, end + 1):
            if track_id not in frame_ids.get(frame_idx, set()):
                issues[track_id].append(frame_idx)

    if not issues:
        print(f"\n {mask_dir} tracks valid")
    else:
        print(f"\nfound {len(issues)} tracks with missing frames:")
        for track_id, missing in sorted(issues.items()):
            print(f"  track {track_id}: missing in frames {missing}")

    return dict(issues)

def patch_missing_tracks(issues, mask_dir, seed=42):
    rng = np.random.default_rng(seed)

    n_files = sum(1 for _ in Path(mask_dir).glob(f"*tif"))
    digits = 4 if n_files > 999 else 3

    for track_id, frames in issues.items():
        for frame_idx in frames:
            mask_path = os.path.join(mask_dir, f"mask{frame_idx:0{digits}d}.tif")
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            h, w = mask.shape
            placed = False
            for j in range(1000):
                y = rng.integers(0, h)
                x = rng.integers(0, w - 2)

                if np.all(mask[y, x:x+3] == 0):
                    mask[y, x:x+3] = track_id
                    placed = True
                    break

            print(f"added 3 pixels for track_id {track_id} in frame {frame_idx}")
            cv2.imwrite(mask_path, mask)

def remap_and_handle_merges(video_segments, inference_state):

    swapped_objs = inference_state.get("_swapped_objs", {})
    old_to_new = {old_id: new_id for new_id, old_id in swapped_objs.items()}
    
    temp_obj_counts = {}
    for frame_idx, frame_data in video_segments.items():
        for obj_id in frame_data.keys():
            if obj_id >= temp_object_start_id and obj_id not in old_to_new:
                if obj_id not in temp_obj_counts:
                    temp_obj_counts[obj_id] = 0
                temp_obj_counts[obj_id] += 1
    
    for temp_id, count in temp_obj_counts.items():
        if count > 0:
            new_id = inference_state["_next_id"]
            inference_state["_next_id"] += 1
            old_to_new[temp_id] = new_id
            swapped_objs[new_id] = temp_id
    
    merge_events = inference_state.get("_merge_event", {})
    
    for frame_idx in merge_events:
        for event in merge_events[frame_idx]:
            if event["parent"] in old_to_new:
                event["parent"] = old_to_new[event["parent"]]
            if event["daughter"] in old_to_new:
                event["daughter"] = old_to_new[event["daughter"]]
    
    for frame_idx, frame_data in video_segments.items():
        remapped_frame = {}
        for obj_id, mask in frame_data.items():
            if obj_id in old_to_new:
                new_id = old_to_new[obj_id]
                remapped_frame[new_id] = mask
            elif obj_id >= temp_object_start_id:
                continue
            else:
                remapped_frame[obj_id] = mask
        video_segments[frame_idx] = remapped_frame

    inference_state["_merge_event"] = merge_events 
    return video_segments, merge_events


def generate_images(video_segments, video_yolo_boxes,input_folder, output_folder, folder):
    os.makedirs(output_folder, exist_ok=True)
    frame_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    np.random.seed(42)
    all_obj_ids = set()
    for seg in video_segments.values():
        all_obj_ids.update(seg.keys())
    num_objects = max(all_obj_ids) + 1
    colors = np.random.randint(50, 255, size=(num_objects, 3), dtype=np.uint8)

    bigdatasets = ["Fluo-N2DL-HeLa", "BF-C2DL-MuSC","PhC-C2DL-PSC","BF-C2DL-HSC"]
    
    for frame_idx in sorted(video_segments.keys()):
        frame_path = os.path.join(input_folder, frame_files[frame_idx])
        frame = np.array(Image.open(frame_path).convert("RGB"))
        overlay = frame.copy()
        for obj_id, mask in video_segments[frame_idx].items():
            color = colors[obj_id]
            overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
    
        img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(img)

        font_size = 24
        if folder in bigdatasets:
             font_size = 12
        
        font = ImageFont.load_default(size=font_size)
    
        yolo_boxes = video_yolo_boxes.get(frame_idx, None)
        if yolo_boxes is not None and len(yolo_boxes) > 0:
            for box_idx, box in enumerate(yolo_boxes):
                x1, y1, x2, y2 = [int(c) for c in box[:4]]
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                if folder not in bigdatasets:
                    draw.text((x1, y1 - 14), str(box_idx), fill=(0, 255, 0), font=font)
        
        for obj_id, mask in video_segments[frame_idx].items():
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            label = str(obj_id)
            color_tuple = tuple(int(c) for c in colors[obj_id])
            for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                draw.text((cx+dx, cy+dy), label, fill=(0,0,0), font=font)
            draw.text((cx, cy), label, fill=color_tuple, font=font)
    
        img.save(os.path.join(output_folder, frame_files[frame_idx]))
    
    print(f"Saved {len(video_segments)} overlay frames to {output_folder}")
    
def generate_ctc_output(video_segments, inference_state, output_folder, frame_shape):
    os.makedirs(output_folder, exist_ok=True)
    next_id = inference_state["_next_id"]
    
    merge_events = inference_state.get("_merge_event", {})
    tracks = []
    active_tracks = {}
    
    frame_relabel = defaultdict(dict)
    sorted_frames = sorted(video_segments.keys())
    
    for frame_idx in sorted_frames:
        current_objs = set()
        for obj_id, mask in video_segments[frame_idx].items():
            if np.sum(mask) > 0:
                current_objs.add(obj_id)
        
        prev_frame = frame_idx - 1
        if prev_frame >= 0 and prev_frame in video_segments:
            prev_objs = set()
            for obj_id, mask in video_segments[prev_frame].items():
                if np.sum(mask) > 0:
                    prev_objs.add(obj_id)
        else:
            prev_objs = set()

        if frame_idx not in frame_relabel and frame_idx != 0:
            frame_relabel[frame_idx] = dict(frame_relabel.get(prev_frame, {}))
        disappeared = prev_objs - current_objs
        
        appeared = current_objs - prev_objs

        parent_tracks = {}
        for merge_event in merge_events.get(prev_frame, []):
            parent = merge_event['parent']
            daughter = merge_event['daughter']
            if parent >= temp_object_start_id or daughter >= temp_object_start_id:
                continue

            if parent in frame_relabel[frame_idx]:
                parent_relabeled = frame_relabel[frame_idx][parent]
            else:
                continue
           
            if parent_relabeled not in active_tracks:
                continue
                
            if active_tracks[parent_relabeled]["start"] == frame_idx:
                if parent in parent_tracks:
                    parent_track = parent_tracks[parent]
                else:
                    continue

            else:
                parent_track = active_tracks.pop(parent_relabeled)
                parent_tracks[parent] = parent_track
                tracks.append((
                    parent_track["track_id"],
                    parent_track["start"],
                    prev_frame,
                    parent_track["parent"],
                ))
                
                parent_mask = video_segments.get(frame_idx, {}).get(parent)
                if parent_mask is not None and np.sum(parent_mask) != 0:
                    if frame_idx > 1600 and frame_idx < 1605 :
                        pass
                        
                    parent_new_id = next_id
                    next_id += 1
                    frame_relabel[frame_idx][parent] = parent_new_id
                    active_tracks[parent_new_id] = {
                        "track_id": parent_new_id,
                        "start": frame_idx,
                        "parent": parent_track["track_id"],
                    }

                    
                if parent in disappeared:
                        disappeared.discard(parent)
                
            daughter_mask = video_segments.get(frame_idx, {}).get(daughter)
            daughter_copy_old = daughter
            
            if daughter_mask is not None and np.sum(daughter_mask) != 0:     
                if daughter in frame_relabel[frame_idx]:

                    daughter_relabeled = frame_relabel[frame_idx][daughter]
                    if daughter_relabeled in active_tracks:
                        daughter_track = active_tracks.pop(daughter_relabeled)
                        tracks.append((
                            daughter_track["track_id"],
                            daughter_track["start"],
                            prev_frame,
                            daughter_track["parent"],
                        ))
                    
                    obj_id_new_id = next_id
                    next_id += 1
                    
                    frame_relabel[frame_idx][daughter_copy_old] = obj_id_new_id
                    daughter = obj_id_new_id
                    
                else:
                    frame_relabel[frame_idx][daughter_copy_old] = daughter
                
                active_tracks[daughter] = {
                    "track_id": daughter,
                    "start": frame_idx,
                    "parent": parent_track["track_id"],
                }
            if daughter_copy_old in appeared:
                    appeared.discard(daughter_copy_old)
            
            
        for obj_id in disappeared:
            obj_id_relabeled = frame_relabel[frame_idx][obj_id]
            if obj_id_relabeled in active_tracks:
                track = active_tracks[obj_id_relabeled]
                tracks.append((
                    track["track_id"],
                    track["start"],
                    prev_frame,
                    track["parent"]
                ))
            del active_tracks[obj_id_relabeled]

        for obj_id in appeared:
            
            obj_mask = video_segments.get(frame_idx, {}).get(obj_id)
            if obj_mask is None or np.sum(obj_mask) == 0:
                continue
                
            parent_track_id = 0
            track_id = obj_id
            
            if obj_id in frame_relabel[frame_idx]:
                parent_track_id = frame_relabel[frame_idx][obj_id]
                obj_id_new_id = next_id
                next_id += 1
                
                frame_relabel[frame_idx][obj_id] = obj_id_new_id
                track_id = obj_id_new_id
            else:
                frame_relabel[frame_idx][obj_id] = obj_id
            
            active_tracks[track_id] = {
                "track_id": track_id,
                "start": frame_idx,
                "parent": parent_track_id,
            }

    last_frame = sorted_frames[-1]
    for obj_id, track in active_tracks.items():
        tracks.append((
            track["track_id"],
            track["start"],
            last_frame,
            track["parent"],
        ))

    tracks.sort(key=lambda x: x[0])
    
    track_path = os.path.join(output_folder, "res_track.txt")
    lines = [f"{L} {B} {E} {P}" for L, B, E, P in tracks]
    with open(track_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved {len(tracks)} tracks to {track_path}")
  
    for frame_idx in sorted_frames:

        mask_combined = np.zeros(frame_shape[:2], dtype=np.uint16)
        
        for obj_id, mask in video_segments[frame_idx].items():
            
            if obj_id != 0 and np.sum(mask) > 0:
                # Apply relabeling if needed
                if obj_id in frame_relabel[frame_idx]:
                     
                    final_id = frame_relabel[frame_idx][obj_id]
                    mask_combined[mask] = final_id

        img = Image.fromarray(mask_combined)
        num = len(sorted_frames)
        if (num < 1000):
            img.save(os.path.join(output_folder, f"mask{frame_idx:03d}.tif"))
        else:
            img.save(os.path.join(output_folder, f"mask{frame_idx:04d}.tif"))

    print(f"Saved {len(video_segments)} mask frames to {output_folder}")


def run_sam2_yolo_pipeline(dataset_folder, track_folder, generate_images_flag=False):
    model_path = "./yolo_best_model//best.pt"
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint)
    predictor.__class__ = SAM2YOLOVideoPredictor
    predictor.match_iou_threshold = 0.4
    predictor.mask_containment_threshold = 0.8
    predictor.temp_object_maturity = 4
    predictor.max_age_without_yolo_box = 6
    predictor.max_frames_retain_removed_obj=16
    predictor.max_swap_distance_threshold=100.0
    predictor.mask_correction_iou_threshold=1.0
    
    detector_fn = make_yolo_detector(
        model_path=model_path,
        detection_conf_threshold=0.4,
    )

    match = re.search(r'/([^/]+)/\d+/?$', dataset_folder)
    dataset_name = match.group(1)
    input_folder = convert_tif_to_jpg(dataset_folder, tmp_name='tmp')

    no_remap_output_folder = f"{dataset_folder}_no_remap_labeled_video"
    remaped_output_folder = f"{dataset_folder}_labeled_video"
    
    inference_state, video_frames = predictor.init_state_with_detector(
        frames_dir=input_folder,
        detector_fn=detector_fn,
        offload_state_to_cpu=False,
    )

    video_segments = {}
    video_yolo_boxes = {}

    for frame_idx, obj_ids, masks, yolo_boxes in predictor.propagate_in_video(
        inference_state,
        detector_fn=detector_fn,
        video_frames=video_frames,
    ):
        masks = (masks > 0.0).cpu().numpy()
        video_segments[frame_idx] = {}


        for i, obj_id in enumerate(obj_ids):
            video_segments[frame_idx][obj_id] = masks[i][0]
            video_yolo_boxes[frame_idx] = yolo_boxes
    
    if generate_images_flag:
        generate_images(video_segments, video_yolo_boxes, input_folder, no_remap_output_folder,dataset_name)
        
    video_segments, merge_events = remap_and_handle_merges(video_segments, inference_state)

    if generate_images_flag:
        generate_images(video_segments, video_yolo_boxes, input_folder, remaped_output_folder, dataset_name)
    
    frame_shape = video_frames[0].shape[:2]
    
    generate_ctc_output(video_segments, inference_state, track_folder, frame_shape)

    issues = validate_res_track(track_folder)
    if issues:
        print("issues found", issues)
    
    remove_tmp_folder(input_folder)
    print(f"Propagation done. Got masks for {len(video_segments)} frames")
    del video_segments, video_yolo_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str)
    parser.add_argument("track_folder", type=str)
    args = parser.parse_args()
    run_sam2_yolo_pipeline(args.dataset_folder, args.track_folder)

