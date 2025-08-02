# orthopedic-surgery-registration


## Installation
Clone the repository:

```bash
git clone https://github.com/xlzhu0317/orthopedic-surgery-registration.git
cd orthopedic-surgery-registration
```

Create conda environment and install requirements:

```bash
conda create -n {environment name} python=3.8
conda activate {environment name}
```

## Requirements
Install packages:

```bash
pip install -r requirements.txt
```
To run this codebase, [PyTorch](https://pytorch.org/get-started/locally/) is required. run the following command:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Install dependencies

```bash
pip install -e .
```


## Data Processing
Download the [dataset](https://drive.google.com/file/d/1M60cFOJw9aDcpTlzWQqfKRFkdXTW2zSu/view?usp=sharing) and unzip to

```bash
./data/dataset
    ├── Distal Femur
    ├── Distal Tibia
    ├── lumbar_vertebral
    ├── Pelvis
    ├── Proximal Femur
    └── Proximal Tibia

```

Generate the data. run the following command:

```bash
python data/data_pkl.py
```


## Usage

### Training
Use the following command for training:

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Test
For testing, use the following command and specify the checkpoint path:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=./checkpoints/best_model.pth.tar --benchmark=test
```
A pretrained model is also available at:

```bash
./checkpoints/best_model.pth.tar
```

### Visualization

To visualize the test results, you can use the following code or add to ./test.py: 

```bash
def visualize_random_results(results_dir, num_to_show=50):
    search_path = osp.join(results_dir, '*.npz')
    all_result_files = glob.glob(search_path)

    num_to_show = min(num_to_show, len(all_result_files))
    selected_files = random.sample(all_result_files, num_to_show)
    

    for file_idx, file_path in enumerate(selected_files):
        scene_name = osp.basename(file_path).replace('.npz', '')
        data = np.load(file_path, allow_pickle=True)
        
        ref_points_loaded = data['ref_points']
        src_points_loaded = data['src_points']
        estimated_transform_loaded = data['estimated_transform']

        if estimated_transform_loaded.ndim == 3:
            estimated_transform_squeezed = np.squeeze(estimated_transform_loaded, axis=0)
        else:
            estimated_transform_squeezed = estimated_transform_loaded

        
        ref_points = np.ascontiguousarray(ref_points_loaded, dtype=np.float64)
        src_points = np.ascontiguousarray(src_points_loaded, dtype=np.float64)
        estimated_transform = estimated_transform_squeezed.astype(np.float64)

        
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
        ref_pcd.paint_uniform_color([1.0, 0, 0])  # red

        src_pcd_original = o3d.geometry.PointCloud()
        src_pcd_original.points = o3d.utility.Vector3dVector(src_points)
        src_pcd_original.paint_uniform_color([0, 0, 1.0])  # blue

        src_pcd_transformed = o3d.geometry.PointCloud(src_pcd_original)
        src_pcd_transformed.transform(estimated_transform)
        src_pcd_transformed.paint_uniform_color([0, 1.0, 0])  # green
        
        o3d.visualization.draw_geometries([ref_pcd, src_pcd_original], window_name=f"Before - {scene_name}")
        o3d.visualization.draw_geometries([ref_pcd, src_pcd_transformed], window_name=f"After - {scene_name}")     
```

It is worth noting that the test results are saved in the ./output as npz format

## Acknowledgements

The code is heavily borrowed from [Geotansformer](https://github.com/qinzheng93/GeoTransformer) [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) [DCP](https://github.com/WangYueFt/dcp/blob/master/model.py).
We thank the authors for their excellent work!
