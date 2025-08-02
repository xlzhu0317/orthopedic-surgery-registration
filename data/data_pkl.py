import os
import os.path as osp
import glob
import pickle
import random
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from config import make_cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_rotation_matrix(max_angle_deg):
    angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
    return R


def process_split(split, cfg, dataset_root, data_save_root):
    overlap_ratios = cfg.data.overlap_ratios
    rot_max_deg = cfg.data.rot_max_deg
    trans_range = cfg.data.trans_range

    entries = []
    for class_dir in sorted(os.listdir(dataset_root)):
        class_path = os.path.join(dataset_root, class_dir)
        if not os.path.isdir(class_path):
            continue
        split_dir = os.path.join(class_path, split)
        if not os.path.isdir(split_dir):
            continue

        ply_files = glob.glob(os.path.join(split_dir, "*.ply"))
        for ply_file in tqdm(ply_files, desc=f"Processing {split} - {class_dir}"):
            pcd = o3d.io.read_point_cloud(ply_file)
            xyz = np.asarray(pcd.points)
            scene_name = os.path.splitext(os.path.basename(ply_file))[0]

            source_save_dir = os.path.join(data_save_root, class_dir, split, "source")
            os.makedirs(source_save_dir, exist_ok=True)
            source_save_path = os.path.join(source_save_dir, f"{scene_name}_source.pth")
            torch.save(xyz.astype(np.float32), source_save_path)

            for overlap in overlap_ratios:
                center_idx = np.random.randint(len(xyz))
                center_point = xyz[center_idx].reshape(1, -1)

                num_neighbors = min(len(xyz) - 1, int(len(xyz) * overlap))
                if num_neighbors < 1: continue

                knn = NearestNeighbors(n_neighbors=num_neighbors)
                knn.fit(xyz)
                indices = knn.kneighbors(center_point, return_distance=False)[0]

                if len(indices) == 0: continue

                sampled_points = xyz[indices]
                target_xyz = np.vstack([center_point, sampled_points])

                R = random_rotation_matrix(rot_max_deg)
                t = np.random.uniform(-trans_range, trans_range, size=(3,))
                transformed_xyz = (R @ target_xyz.T).T + t

                if overlap == 1.0:
                    name_suffix = ""
                else:
                    name_suffix = f"_{int(overlap * 10)}"

                target_save_dir = os.path.join(data_save_root, class_dir, split, "target")
                os.makedirs(target_save_dir, exist_ok=True)

                target_save_path = os.path.join(target_save_dir, f"{scene_name}_target{name_suffix}.pth")
                torch.save(transformed_xyz.astype(np.float32), target_save_path)

                inv_R = R.T
                inv_t = - R.T @ t


                entry = {
                    "overlap": overlap,
                    "pcd0": os.path.relpath(source_save_path, data_save_root),
                    "pcd1": os.path.relpath(target_save_path, data_save_root),
                    "rotation": inv_R.astype(np.float32),
                    "translation": inv_t.astype(np.float32),
                    "scene_name": f"{scene_name}{name_suffix}",
                }
                entries.append(entry)
    return entries


def main():
    cfg = make_cfg()
    set_seed(cfg.seed)

    dataset_root = osp.join(cfg.data.dataset_root, 'dataset')
    data_save_root = osp.join(cfg.data.dataset_root, 'data')
    metadata_root = osp.join(cfg.data.dataset_root, 'metadata')

    print("开始生成数据集...")
    os.makedirs(metadata_root, exist_ok=True)
    for split in ["train", "val", "test"]:
        data = process_split(split, cfg, dataset_root, data_save_root)
        pkl_path = os.path.join(metadata_root, f"{split}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

    print(f"\n 数据生成完毕！")

if __name__ == "__main__":
    main()