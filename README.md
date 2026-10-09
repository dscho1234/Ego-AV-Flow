# EgoAVFlow

**EgoAVFlow: Robot Policy Learning with Active Vision from Human Egocentric Videos via 3D Flow**

[[Project Page]](https://dscho1234.github.io/egoavflow/) [[arXiv]](https://arxiv.org/abs/2602.22461)

EgoAVFlow learns robot manipulation and active viewpoint control from egocentric human videos through a shared 3D flow representation. It consists of three diffusion models:

| Component | Config key | Predicts |
| --- | --- | --- |
| Robot policy | `model` | future robot actions |
| Flow generation model | `flow_model` | future 3D flow of the query points |
| View policy | `view_model` | future camera viewpoints |

At test time, the view policy is refined with reward-maximizing denoising under a visibility-aware reward computed from the predicted robot motion, the predicted 3D flow, and the reconstructed scene geometry.

This repository contains the code to **preprocess** human demonstrations into the training data, to **train** the three models, and to **evaluate a trained model offline** on recorded episodes (no robot hardware required).

## Repository layout

```
.
├── config/
│   ├── active_vision/
│   │   ├── train_active_vision.yaml      # training
│   │   ├── evaluate_active_vision.yaml   # offline evaluation
│   │   └── paths/default.yaml            # data / output / checkpoint locations
│   └── preprocessing/                    # one config per preprocessing step (+ common.yaml)
├── egoavflow/
│   ├── models/                           # diffusion transformer (CustomRDTRunner)
│   ├── diffusion_policy/dataloader/      # zarr replay buffer and dataset
│   ├── diffusion_policy/utility/         # offline evaluation loop
│   ├── preprocessing/                    # episode I/O and DIFT features for the preprocessing scripts
│   ├── svdd.py                           # visibility reward and reward-maximizing denoising
│   ├── visibility.py                     # nvblox scene mesh, Z1 arm model, raycasting
│   ├── widowx_visualizer.py              # WidowX AI arm model (FK / IK / meshes)
│   └── common/                           # EMA, zarr codecs, I/O and visualization helpers
├── scripts/
│   ├── active_vision/                    # train_active_vision.py, evaluate_active_vision.py
│   ├── preprocessing/                    # data preprocessing pipeline (see below)
│   ├── download_checkpoints.sh           # pretrained third-party weights
│   └── install_preprocessing.sh          # builds the preprocessing dependencies
├── example_data/                         # one processed episode and its raw recording (see "Example episode")
├── assets/robots/                        # Unitree Z1 and Trossen WidowX AI descriptions (URDF + meshes)
└── third_party/
    ├── cotracker3/                       # CoTracker3 with per-frame online updates (modified copy)
    ├── hamer/                            # git submodules used by the preprocessing
    ├── DROID-SLAM/
    ├── Cutie/
    └── Grounded-Segment-Anything/
```

## Installation

Tested with Python 3.10, PyTorch 2.5.1 (CUDA 12.1) on Ubuntu 22.04 and 24.04 with NVIDIA A40 GPUs.

```bash
git clone https://github.com/dscho1234/Ego-AV-Flow.git egoavflow
cd egoavflow
conda env create -f environment.yml
conda activate egoavflow
pip install -e .
```

`pip install -e .` makes the `egoavflow` package importable from the scripts. Training only needs the steps above. The submodules under `third_party/` are only needed for the [data preprocessing](#data-preprocessing).

### Additional dependencies for evaluation

Offline evaluation reconstructs the scene with nvblox, tracks query points with CoTracker3, and renders the two robot arms from their URDFs (included in `assets/robots/`).

**nvblox_torch**

```bash
pip install https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.8/nvblox_torch-0.0.8rc5+cu11ubuntu22-863-py3-none-linux_x86_64.whl
```

This wheel links against the CUDA 11 runtime and NPP libraries (`libcudart.so.11.0`, `libnpp*.so.11`). If they are not available on your system, install them into the environment and put `$CONDA_PREFIX/lib` on `LD_LIBRARY_PATH`:

```bash
conda install -c nvidia cuda-cudart=11.8 libnpp=11.8
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Pick another wheel from the [nvblox releases](https://github.com/nvidia-isaac/nvblox/releases) if your CUDA / Ubuntu versions differ.

**CoTracker3 (online, per-frame update)**

The evaluation advances the online tracker one frame at a time over a sliding window (`CoTrackerOnlinePredictor(video_chunk, one_frame=True)`). This option is not part of the upstream [CoTracker3](https://github.com/facebookresearch/co-tracker) API; `third_party/cotracker3` is a copy of CoTracker3 (commit `b00a83b`) with this option added. Install it and download its checkpoint:

```bash
pip install -e third_party/cotracker3
bash scripts/download_checkpoints.sh eval    # -> checkpoints/cotracker3/scaled_online.pth
```

**Robot descriptions**: the Unitree Z1 (camera arm, `assets/robots/z1_description`) and the Trossen WidowX AI (manipulation arm, `assets/robots/trossen_arm_description`) URDFs and meshes used by the evaluation are included in the repository.

## Paths

Machine-specific locations are defined in `config/active_vision/paths/default.yaml` and can be set with environment variables or Hydra overrides:

| Config key | Environment variable | Default | Contents |
| --- | --- | --- | --- |
| `paths.data_root` | `EGOAVFLOW_DATA_ROOT` | `<repo>/data` | processed datasets |
| `paths.output_root` | `EGOAVFLOW_OUTPUT_ROOT` | `<repo>/outputs` | training runs |
| `paths.checkpoint_root` | `EGOAVFLOW_CHECKPOINT_ROOT` | `<repo>/checkpoints` | pretrained third-party weights (`scripts/download_checkpoints.sh`) |

The preprocessing configs (`config/preprocessing/common.yaml`) use the same `EGOAVFLOW_CHECKPOINT_ROOT` and `EGOAVFLOW_OUTPUT_ROOT` (Hydra logs).

When set through the environment variables, these values stay interpolations in the saved run config, so a trained model resolves them again on the machine where it is evaluated (values passed on the command line are stored as-is).

## Data

An example episode is included in `example_data/` (see [Example episode](#example-episode)). The [data preprocessing](#data-preprocessing) turns RealSense recordings into the following format.

Each task is stored as three zarr groups under `paths.data_root`:

```
data/
├── <task>/              # training episodes
├── <task>_invisible/    # additional training episodes (a separate recording of the same task)
└── <task>_validation/   # held-out episodes (validation loss, offline evaluation)
```

Each zarr group contains `episode_0`, `episode_1`, ... with (T = number of frames):

| Key | Shape | Description |
| --- | --- | --- |
| `action`, `proprioception` | `(T, 10)` | hand-derived gripper position, 6D rotation, and binary gripper state (camera frame) |
| `T_mc_opt_droid` | `(T, 4, 4)` | camera pose in the ChArUco marker frame |
| `dift_points` | `(N, 2)` | query points in the first frame (640x480 pixels) |
| `dift_point_tracking_sequence` | `(N, T, 4)` | 3D tracks of the query points (xyz in the camera frame, visibility) |
| `dift_pixel_tracking_sequence` | `(N, T, 3)` | 2D tracks of the query points (uv in the 256x256 tracking image, visibility) |
| `mask_point_tracking_sequence`, `mask_pixel_tracking_sequence` | `(K, M, T, 4)`, `(K, M, T, 3)` | 3D / 2D tracks of M points sampled on each of the K object masks (read with `dataset.use_dift_point_tracking=false`) |
| `task_description` | `(1,)` | language description of the task |
| `camera_0/rgb`, `camera_0/depth` | `(T, H, W, 3)`, `(T, H, W)` | RGB-D frames (depth in mm), evaluation only |
| `intrinsics` | `(3, 3)` | camera intrinsics, evaluation only |
| `sam_mask_sequence_multi_obj` | `(T, K, H, W)` | object masks followed by the hand mask, evaluation only |

When the data is loaded, actions, proprioception, and tracks are converted to the marker frame (`dataset.use_marker_coordinate=True`) and temporally downsampled by `dataset.downsample_rate` (2). RGB frames are stored with the JPEG XL codec registered in `egoavflow/common/imagecodecs_numcodecs.py`.

### Example episode

`example_data/` contains one episode of the bottle task to try the code:

- `example_data/charuco_marker_bottle_under_table_validation/`: the processed episode in the format above (321 frames, 28 MB).
- `example_data/raw/charuco_marker_bottle_under_table_validation/`: its raw recording (the output of preprocessing step 1: 400 RGB-D frames as PNG files, 216 MB) and the clicks of steps 3 and 6 (`segmentation_clicks.json`, `query_points.json`). The PNG files are stored with [Git LFS](https://git-lfs.com) and are not downloaded by `git clone`; see [below](#preprocessing-the-example).

#### Training and evaluation

From the repository root:

```bash
export WANDB_MODE=disabled
# train on the example episode (config/active_vision/train_example.yaml)
python scripts/active_vision/train_active_vision.py --config-name train_example
# offline evaluation of the resulting model on the first 20 steps (needs the evaluation dependencies)
python scripts/active_vision/evaluate_active_vision.py \
    model_path=$PWD/outputs/active_vision/charuco_marker_bottle_under_table/example \
    ckpt=99 downsample_ratio=2 max_steps=20
```

On an NVIDIA A40, training (100 epochs) takes about 2 minutes and writes about 4 GB of checkpoints (epochs 0 and 99), and the evaluation of 20 steps with the three reward types takes about 8 minutes. A model trained on a single episode is not useful; the run only exercises the training and evaluation code. If `EGOAVFLOW_OUTPUT_ROOT` is set, the run directory is under it instead of `outputs/`.

#### Preprocessing the example

With the [preprocessing environment](#data-preprocessing) and Git LFS installed, download the raw recording and run steps 2 to 8 on a copy of it:

```bash
git lfs pull --include="example_data/raw/**" --exclude=""
cp -r example_data/raw/charuco_marker_bottle_under_table_validation /tmp/egoavflow_example
D=/tmp/egoavflow_example
cd scripts/preprocessing
python estimate_hand_pose.py "data_dirs=[$D]"
python build_episodes.py "data_dirs=[$D]" "task=move the spray" "object=[spray]"
python estimate_camera_trajectory.py "data_dirs=[$D]"
python align_marker_frame.py "data_dirs=[$D]"
python annotate_query_points.py "data_dirs=[$D]"
python generate_flow.py "data_dirs=[$D]"
python plot_marker_trajectory.py "data_dirs=[$D]"
```

Steps 3 and 6 reuse the saved clicks and do not open a window (delete the two JSON files or pass `reuse_clicks=false` to click yourself). On an NVIDIA A40 the steps take about 11 minutes, most of it for the hand pose. The result matches the processed episode in `example_data/`: same start frame, RGB, depth, and query points; gripper trajectory within 1 mm; mask IoU above 0.99; and camera poses in the marker frame within about 1 cm, the run-to-run variation of DROID-SLAM.

## Data preprocessing

The scripts in `scripts/preprocessing/` turn RealSense recordings of human demonstrations into the format above. They run in a separate environment (DIFT needs diffusers 0.15, HaMeR needs mmcv 1.3.9 and detectron2) and need a CUDA GPU.

### Setup

```bash
git submodule update --init third_party/Cutie third_party/Grounded-Segment-Anything
git submodule update --init --recursive third_party/hamer third_party/DROID-SLAM
conda env create -f environment_preprocess.yml
conda activate egoavflow-preprocess
bash scripts/install_preprocessing.sh
bash scripts/download_checkpoints.sh preprocess
```

Initialize only these four submodules: a top-level `git submodule update --init --recursive` also clones two large submodules of Grounded-Segment-Anything that are not used. `install_preprocessing.sh` installs HaMeR, ViTPose, GroundingDINO, Segment Anything, Cutie, and DROID-SLAM from the submodules and compiles their CUDA extensions for `TORCH_CUDA_ARCH_LIST` (default `7.5;8.0;8.6;8.9;9.0`), so it can run on a machine without a GPU.

HaMeR uses the MANO hand model, which cannot be redistributed: register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), download the model, and copy `MANO_RIGHT.pkl` to `third_party/hamer/_DATA/data/mano/`.

`download_checkpoints.sh preprocess` downloads:

| Model | Used by | Location |
| --- | --- | --- |
| HaMeR, ViTPose-H | hand pose | `third_party/hamer/_DATA/` |
| GroundingDINO Swin-T, SAM ViT-H | segmentation | `checkpoints/grounding_dino/`, `checkpoints/sam/` |
| Cutie | mask propagation | `third_party/Cutie/weights/` |
| DROID-SLAM | camera trajectory | `checkpoints/droid/` |
| Stable Diffusion 2.1 | DIFT query point transfer | `checkpoints/stable-diffusion-2-1/` |
| CoTracker3 (online) | point tracking | `checkpoints/cotracker3/` |

The original `stabilityai/stable-diffusion-2-1` repository is no longer available on Hugging Face, so Stable Diffusion 2.1 is downloaded from the [`sd2-community/stable-diffusion-2-1`](https://huggingface.co/sd2-community/stable-diffusion-2-1) mirror. The ViTDet person detector (used by HaMeR) and the BERT tokenizer of GroundingDINO are downloaded automatically on first use.

### Pipeline

Run the steps below from `scripts/preprocessing/` for every recording directory of a task (`<task>`, `<task>_invisible`, `<task>_validation` are separate recordings). Except for the recorder, the scripts are Hydra scripts configured in `config/preprocessing/` (shared settings, including the camera intrinsics, are in `common.yaml`); they take `data_dirs=[...]` and optionally `episode_start` / `episode_end`. Example for the bottle task:

```bash
cd scripts/preprocessing
D=$EGOAVFLOW_DATA_ROOT/charuco_marker_bottle_under_table
python record_realsense.py --out_dir $D --num_episodes 100                    # 1. record
python estimate_hand_pose.py "data_dirs=[$D]"                                  # 2. hand pose
python build_episodes.py "data_dirs=[$D]" "task=move the spray" "object=[spray]"  # 3. masks, actions
python estimate_camera_trajectory.py "data_dirs=[$D]"                          # 4. DROID-SLAM
python align_marker_frame.py "data_dirs=[$D]"                                  # 5. ChArUco frame
python annotate_query_points.py "data_dirs=[$D]"                               # 6. click query points
python generate_flow.py "data_dirs=[$D]"                                       # 7. 2D / 3D flow
python plot_marker_trajectory.py "data_dirs=[$D]"                              # 8. visual check
python renumber_episodes.py "data_dirs=[$D]"                                   # 9. after deleting bad episodes
```

1. **Recording** (`record_realsense.py`). An egocentric Intel RealSense D435 records aligned RGB-D at 640x480 and 30 Hz, with a ChArUco board placed in the scene, into `episode_<i>/color/` and `episode_<i>/depth/`. Each recording starts with 100 frames of camera motion before the demonstration (`--num_lag_steps`), which are used only for SLAM.
2. **Hand pose** (`estimate_hand_pose.py`). For every frame, the person is detected with ViTDet, the right-hand keypoints with ViTPose, and the MANO hand with [HaMeR](https://github.com/geopavlakos/hamer). The wrist pose is refined by minimizing the reprojection error of the 21 joints, and its metric translation is recovered from the depth at the wrist (plus 3 cm for the wrist thickness). A parallel-gripper frame is attached to the hand, with its origin between the thumb and index-finger bases and its axes from a plane fit to the thumb and index joints. Writes `episode_<i>/data_dict.pkl` and an overlay video.
3. **Segmentation and episodes** (`build_episodes.py`). For every episode, a window shows the recording: pick the first frame in which all task objects are visible, click one point per object (in the order of `object`), confirm with `c`, then do the same for the hand. SAM turns the clicks into masks, and [Cutie](https://github.com/hkchengrex/Cutie) propagates them (`sam_mask_sequence_multi_obj`: object masks followed by the hand mask). The clicks of all episodes are collected first; the rest runs unattended. The clicks are saved to `episode_<i>/segmentation_clicks.json` and reused when the step is run again, without opening the window (`reuse_clicks=false` to click again). With `manual_segmentation=false`, [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) boxes are used instead of clicks. The episode is clipped to start at the object frame; the hand trajectory is interpolated to every frame (SLERP for rotations), extrapolated backwards if the hand appears later, smoothed, and stored as `action` / `proprioception`. The gripper is closed when the thumb tip and the index tip are closer than `grasp.max_thumb_index_dist`. To change only this threshold later, rerun with `update_gripper_only=true` (this recomputes the gripper state from the stored hand trajectory, extrapolated backwards from the first HaMeR detection instead of the clicked hand frame). Settings of the tasks in the paper:

   | Task | `task` | `object` | `grasp.max_thumb_index_dist` |
   | --- | --- | --- | --- |
   | `charuco_marker_bottle_under_table` | `move the spray` | `[spray]` | 0.06 |
   | `charuco_marker_box_under_drawer` | `move the puppy doll` | `[puppy doll,rattan basket]` | 0.06 |
   | `charuco_marker_toilet_paper_under_drawer` | `move the toilet paper` | `[toilet paper]` | 0.08 |
   | `charuco_marker_towel` | `move the towel` | `[brown towel,rattan basket]` | 0.05 (set with `update_gripper_only=true`) |

4. **Camera trajectory** (`estimate_camera_trajectory.py`). RGB-D [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) runs on the full recording. The manipulated (first) object and the hand are hidden in the episode frames (random RGB noise, zero depth), so that only the static scene drives the camera estimate.
5. **Marker frame** (`align_marker_frame.py`). The ChArUco board (3x5 squares of 5.2 cm, 4.16 cm markers, `DICT_4X4_250`, reference marker id 1) is detected in every frame, and a [GTSAM](https://github.com/borglab/gtsam) pose graph combines the DROID-SLAM odometry with the marker detections into the camera pose in the marker frame, `T_mc_opt_droid`.
6. **Query points** (`annotate_query_points.py`). Click the query points (6 in the paper) on the first frame of episode 0 of each directory, in the same order in every directory. The first 4 lie on the manipulated object (`dataset.num_points_for_object`); with a second object, the remaining points lie on it (`query_point_object_ranges` in `flow.yaml`). Like the segmentation clicks, the points are saved to `episode_0/query_points.json` and reused.
7. **2D and 3D flow** (`generate_flow.py`). The query points are transferred to the first frame of every other episode by [DIFT](https://github.com/Tsingularity/dift) correspondence on Stable Diffusion 2.1 features, tracked with online [CoTracker3](https://github.com/facebookresearch/co-tracker) at 256x256 (`dift_pixel_tracking_sequence`), and lifted to 3D with the recorded depth in the camera frame (`dift_point_tracking_sequence`). A point is marked invisible when its depth deviates by more than 10 cm from the mean depth of its object mask. Points sampled on the object masks are tracked the same way (`mask_{pixel,point}_tracking_sequence`).
8. **Check** (`plot_marker_trajectory.py`) writes an interactive plot of all gripper trajectories and query point tracks in the marker frame (`<data_dir>/marker_frame_trajectories.html`).
9. **Cleanup** (`renumber_episodes.py`). Delete the folders of invalid episodes (e.g. failed hand or object segmentation, wrong tracks), then renumber the remaining episodes to `0, 1, ...`.

Steps 3 and 6 open windows and need a display, unless their clicks have been saved before. The other steps can be distributed over several GPUs, e.g. `python run_multi_gpu.py --gpus 0,1,2,3 estimate_camera_trajectory.py "data_dirs=[$D]"`. The random parts (the masking noise of step 4, the DIFT noise and the mask point sampling of step 7) are seeded per episode (`seed`), so rerunning a step gives the same result.

## Training

```bash
cd scripts/active_vision
python train_active_vision.py dataset_name=charuco_marker_bottle_under_table
```

Multi-GPU training uses [Accelerate](https://huggingface.co/docs/accelerate):

```bash
accelerate launch train_active_vision.py dataset_name=charuco_marker_bottle_under_table
```

Runs are written to `${paths.output_root}/active_vision/<task>/<exp_name>/`, which contains the run config (`.hydra/`), normalization statistics (`stats.pickle`), model checkpoints (`checkpoints/epoch_<k>{,_view,_flow}.ckpt` and their `ema_` versions), and accelerator states (`state/`). Training is logged to [Weights & Biases](https://wandb.ai) (project `active_vision_policy_3d`); set `WANDB_MODE=offline` or `WANDB_MODE=disabled` to run without an account.

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `training.epochs` | `3001` | number of epochs; checkpoints are saved every `(epochs - 1) / 4` epochs |
| `training.batch_size` | `128` | batch size |
| `training.condition_ratio` | `0.8` | probability of conditioning the view policy on flow and viewpoint (otherwise trained unconditionally) |
| `training.use_pixel_flow_for_robot_policy` | `false` | condition the models on 2D pixel flow instead of 3D flow |
| `dataset.pred_horizon` / `dataset.obs_horizon` | `24` / `3` | prediction and observation horizons |
| `eval_off` | `False` | skip the offline evaluation that is launched after each checkpoint |
| `debug` | `False` | load 5 episodes and checkpoint / evaluate every epoch |

Every `evaluation.eval_frequency` epochs the training loop launches `evaluate_active_vision.py` on the latest checkpoint (requires the [evaluation dependencies](#additional-dependencies-for-evaluation)); pass `eval_off=true` to disable this.

## Evaluation

The offline evaluation replays recorded episodes: at every step it tracks the query points with CoTracker3, lifts them to 3D with the recorded depth and camera poses, predicts robot actions and future 3D flow, builds the scene mesh with nvblox, and samples camera viewpoints from the view policy with visibility-aware reward-maximizing denoising.

```bash
cd scripts/active_vision
python evaluate_active_vision.py \
    model_path=/absolute/path/to/run_directory \
    ckpt=3000 \
    downsample_ratio=2
```

`model_path` must be an absolute path (Hydra changes the working directory to the evaluation output directory). `downsample_ratio` must match `dataset.downsample_rate` used for training (`2` by default). `max_steps=<k>` evaluates only the first `k` steps of each episode. The 3D visualization videos are rendered with Open3D windows and need a display; on a headless machine they are skipped with a warning (run under a virtual display, e.g. `xvfb-run -a python evaluate_active_vision.py ...`, to record them), and everything else is still computed and saved. By default, the first episode (`evaluation.gt_eval_num` in the training config) of each data directory in the run's `evaluation.gt_eval_dataset_args.data_dirs` is evaluated; use `data_dirs='[/path/to/<task>_validation]'` to evaluate other data. The camera / robot calibration of the recording setup (`T_B_M`, `T_B_M_view`, `T_E_C_view`, `ee_bias`) is defined in `config/active_vision/evaluate_active_vision.yaml`.

Results are written to `<model_path>/evaluation/epoch_<ckpt>/gt_flow(...)/buffer_<i>/`, one `buffer_<i>` per data directory. Each episode is evaluated with three reward types (`visibility`, `visibility+close_to_nominal`, `visibility+close_to_query_points`):

- `visualization_<k>_<reward type>_mesh.mp4`: scene mesh, robots, and sampled camera frustums with lines of sight,
- `reward_<reward type>/episode_<k>_flow_policy_3d.mp4`: tracked query points,
- `reward_<reward type>/episode_<k>_3d_visualization.ply`: reconstructed scene mesh,
- `reward_<reward type>/action_flow_comparison_episode_<k>.pkl`, `reward_<reward type>/visualization_data_episode_<k>.pkl`: predictions, ground truth, and per-step visibility rewards.

## Acknowledgements

This code base started from [Im2Flow2Act](https://github.com/real-stanford/im2flow2act). The diffusion transformer is adapted from [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer) (which builds on [DiT](https://github.com/facebookresearch/DiT)). We also use [CoTracker3](https://github.com/facebookresearch/co-tracker), [nvblox](https://github.com/nvidia-isaac/nvblox), [diffusers](https://github.com/huggingface/diffusers), and [imagecodecs](https://github.com/cgohlke/imagecodecs), and, for the data preprocessing, [HaMeR](https://github.com/geopavlakos/hamer), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [detectron2](https://github.com/facebookresearch/detectron2), [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) (GroundingDINO and Segment Anything), [Cutie](https://github.com/hkchengrex/Cutie), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [GTSAM](https://github.com/borglab/gtsam), [DIFT](https://github.com/Tsingularity/dift), and [Stable Diffusion 2.1](https://github.com/Stability-AI/stablediffusion).

## Citation

```bibtex
@article{cho2026egoavflow,
  title   = {EgoAVFlow: Robot Policy Learning with Active Vision from Human Egocentric Videos via 3D Flow},
  author  = {Cho, Daesol and Jang, Youngseok and Xu, Danfei and Ha, Sehoon},
  journal = {arXiv preprint arXiv:2602.22461},
  year    = {2026}
}
```

## License

The code and the data, including the example episode in `example_data/`, are released under the MIT license (see [LICENSE](LICENSE)). The third-party components below keep their own licenses:

| Component | Origin | License |
| --- | --- | --- |
| `third_party/cotracker3/` | [CoTracker3](https://github.com/facebookresearch/co-tracker) (commit `b00a83b`), modified for per-frame online updates | CC BY-NC 4.0 |
| `assets/robots/z1_description/` | [unitree_ros](https://github.com/unitreerobotics/unitree_ros) (URDF and collision meshes) | BSD 3-Clause |
| `assets/robots/trossen_arm_description/` | [trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description) (WidowX AI URDF and meshes) | BSD 3-Clause |
| `egoavflow/preprocessing/dift.py` | adapted from [DIFT](https://github.com/Tsingularity/dift) | MIT |
| `third_party/{hamer,DROID-SLAM,Cutie,Grounded-Segment-Anything}` | git submodules, unmodified | see each repository |

The pretrained weights downloaded by `scripts/download_checkpoints.sh` and the MANO model are subject to the licenses of their providers.
