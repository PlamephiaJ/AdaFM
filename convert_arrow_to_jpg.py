import argparse
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data as data_utils
from datasets import load_dataset
from tqdm import tqdm
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torchvision.models.inception import Inception_V3_Weights, inception_v3


def _safe_label_name(hf_ds, label_id: int) -> str:
	try:
		label_name = hf_ds.features["label"].int2str(label_id)
		return label_name.replace("/", "_")
	except Exception:
		return f"class_{label_id:04d}"


def convert_split(
	hf_dataset_id: str,
	split: str,
	cache_dir: str | None,
	output_dir: Path,
	max_images: int | None,
	num_proc: int,
) -> None:
	hf_ds = load_dataset(hf_dataset_id, split=split, cache_dir=cache_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	label_names = None
	if "label" in hf_ds.features:
		try:
			label_names = hf_ds.features["label"].names
		except Exception:
			label_names = None

	if label_names:
		for name in label_names:
			(output_dir / name.replace("/", "_")).mkdir(parents=True, exist_ok=True)

	if max_images is not None:
		hf_ds = hf_ds.select(range(min(len(hf_ds), max_images)))

	def save_batch(batch, indices):
		images = batch["image"]
		labels = batch.get("label", [0] * len(images))
		for img, label, idx in zip(images, labels, indices):
			image = img.convert("RGB")
			label_name = (
				label_names[int(label)].replace("/", "_")
				if label_names
				else f"class_{int(label):04d}"
			)
			label_dir = output_dir / label_name
			label_dir.mkdir(parents=True, exist_ok=True)
			image_path = label_dir / f"{idx:08d}.jpg"
			image.save(image_path, format="JPEG", quality=95, optimize=True)
		return {}

	hf_ds.map(
		save_batch,
		with_indices=True,
		batched=True,
		batch_size=128,
		num_proc=num_proc,
		load_from_cache_file=False,
		desc=f"Converting {split}",
	)


def _get_inception_model(device: torch.device) -> torch.nn.Module:
	model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
	model.fc = torch.nn.Identity()
	model.eval().to(device)
	return model


def _get_inception_transform(image_size: int = 299) -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize(image_size),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		]
	)


def _update_running_stats(mean: np.ndarray, m2: np.ndarray, n: int, batch: np.ndarray):
	if batch.size == 0:
		return mean, m2, n

	batch_n = batch.shape[0]
	batch_mean = batch.mean(axis=0)
	batch_centered = batch - batch_mean
	batch_m2 = batch_centered.T @ batch_centered

	if n == 0:
		return batch_mean, batch_m2, batch_n

	delta = batch_mean - mean
	total_n = n + batch_n
	new_mean = mean + delta * (batch_n / total_n)
	m2 = m2 + batch_m2 + np.outer(delta, delta) * (n * batch_n / total_n)
	return new_mean, m2, total_n


def _merge_stats(
	mean_a: np.ndarray,
	m2_a: np.ndarray,
	n_a: int,
	mean_b: np.ndarray,
	m2_b: np.ndarray,
	n_b: int,
):
	if n_a == 0:
		return mean_b, m2_b, n_b
	if n_b == 0:
		return mean_a, m2_a, n_a

	delta = mean_b - mean_a
	total_n = n_a + n_b
	mean = mean_a + delta * (n_b / total_n)
	m2 = m2_a + m2_b + np.outer(delta, delta) * (n_a * n_b / total_n)
	return mean, m2, total_n


def _compute_shard_stats(
	train_dir: Path,
	stats_path: Path,
	gpu_id: int,
	shard_indices: list[int],
	num_workers: int,
):
	torch.cuda.set_device(gpu_id)
	device = torch.device(f"cuda:{gpu_id}")
	dataset = tv_datasets.ImageFolder(
		root=str(train_dir), transform=_get_inception_transform(299)
	)
	shard = data_utils.Subset(dataset, shard_indices)
	loader = data_utils.DataLoader(
		shard, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True
	)

	model = _get_inception_model(device)
	mean = None
	m2 = None
	n = 0

	with torch.no_grad():
		for images, _ in loader:
			images = images.to(device)
			feats = model(images).detach().cpu().numpy()
			if mean is None:
				mean = np.zeros(feats.shape[1], dtype=np.float64)
				m2 = np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
			mean, m2, n = _update_running_stats(mean, m2, n, feats)

	stats_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez(str(stats_path), mu=mean, m2=m2, n=n)


def precache_fid_stats(
	train_dir: Path,
	stats_path: Path,
	cuda: bool,
	num_workers: int,
	gpu_ids: list[int] | None = None,
) -> None:
	if not train_dir.exists():
		raise FileNotFoundError(f"Train directory not found: {train_dir}")

	if cuda and gpu_ids and len(gpu_ids) > 1:
		dataset = tv_datasets.ImageFolder(
			root=str(train_dir), transform=_get_inception_transform(299)
		)
		indices = list(range(len(dataset)))
		shards = [indices[i:: len(gpu_ids)] for i in range(len(gpu_ids))]

		ctx = mp.get_context("spawn")
		procs = []
		part_paths = []
		for rank, gpu_id in enumerate(gpu_ids):
			part_path = stats_path.with_suffix(f".part{rank}.npz")
			part_paths.append(part_path)
			p = ctx.Process(
				target=_compute_shard_stats,
				args=(
					train_dir,
					part_path,
					gpu_id,
					shards[rank],
					max(1, num_workers // len(gpu_ids)),
				),
			)
			p.start()
			procs.append(p)

		for p in procs:
			p.join()

		mean = None
		m2 = None
		n = 0
		for part_path in part_paths:
			part = np.load(part_path)
			part_mean = part["mu"]
			part_m2 = part["m2"]
			part_n = int(part["n"])
			if mean is None:
				mean = np.zeros_like(part_mean, dtype=np.float64)
				m2 = np.zeros_like(part_m2, dtype=np.float64)
			mean, m2, n = _merge_stats(mean, m2, n, part_mean, part_m2, part_n)

		if n < 2:
			raise ValueError("Not enough samples to compute covariance.")

		cov = m2 / (n - 1)
		stats_path.parent.mkdir(parents=True, exist_ok=True)
		np.savez(str(stats_path), mu=mean, sigma=cov)
		for part_path in part_paths:
			part_path.unlink(missing_ok=True)
		return

	device = torch.device("cuda" if cuda else "cpu")
	dataset = tv_datasets.ImageFolder(
		root=str(train_dir), transform=_get_inception_transform(299)
	)
	loader = data_utils.DataLoader(
		dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True
	)

	model = _get_inception_model(device)
	mean = None
	m2 = None
	n = 0

	with torch.no_grad():
		for images, _ in tqdm(loader, desc="Extracting Inception features"):
			images = images.to(device)
			feats = model(images).detach().cpu().numpy()
			if mean is None:
				mean = np.zeros(feats.shape[1], dtype=np.float64)
				m2 = np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
			mean, m2, n = _update_running_stats(mean, m2, n, feats)

	if n < 2:
		raise ValueError("Not enough samples to compute covariance.")

	cov = m2 / (n - 1)
	stats_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez(str(stats_path), mu=mean, sigma=cov)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert HF ImageNet-1k arrow data to JPG and pre-cache FID stats."
	)
	parser.add_argument(
		"--hf-dataset",
		type=str,
		default="ILSVRC/imagenet-1k",
		help="Hugging Face dataset id",
	)
	parser.add_argument(
		"--cache-dir",
		type=str,
		default="data/imagenet",
		help="HF cache directory (arrow files)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="data/imagenet/jpg",
		help="Output directory for JPGs",
	)
	parser.add_argument(
		"--train-split",
		type=str,
		default="train",
		help="HF train split name",
	)
	parser.add_argument(
		"--val-split",
		type=str,
		default="validation",
		help="HF validation split name",
	)
	parser.add_argument(
		"--max-images",
		type=int,
		default=None,
		help="Optional limit per split for quick tests",
	)
	parser.add_argument(
		"--num-proc",
		type=int,
		default=64,
		help="Number of processes for conversion",
	)
	parser.add_argument(
		"--fid-num-workers",
		type=int,
		default=16,
		help="DataLoader workers for FID stats extraction",
	)
	parser.add_argument(
		"--skip-val",
		action="store_true",
		help="Skip converting validation split",
	)
	parser.add_argument(
		"--skip-convert",
		action="store_true",
		default=True,
		help="Skip JPG conversion and only run FID cache",
	)
	parser.add_argument(
		"--no-fid-cache",
		action="store_true",
		help="Skip pre-caching FID stats for train set",
	)
	parser.add_argument(
		"--cuda",
		default=True,
		action="store_true",
		help="Use CUDA for FID pre-cache",
	)
	parser.add_argument(
		"--gpu-ids",
		type=str,
		default="0,1",
		help="Comma-separated GPU ids for multi-GPU FID cache, e.g. 0,1,2,3",
	)

	args = parser.parse_args()

	cache_dir = args.cache_dir if args.cache_dir else None
	output_dir = Path(args.output_dir)

	train_dir = output_dir / "train"
	if not args.skip_convert:
		convert_split(
			args.hf_dataset,
			args.train_split,
			cache_dir,
			train_dir,
			args.max_images,
			args.num_proc,
		)

		if not args.skip_val:
			val_dir = output_dir / "val"
			convert_split(
				args.hf_dataset,
				args.val_split,
				cache_dir,
				val_dir,
				args.max_images,
				args.num_proc,
			)

	if not args.no_fid_cache:
		stats_path = output_dir / "imagenet_fid_stats_train.npz"
		gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip().isdigit()]
		precache_fid_stats(
			train_dir,
			stats_path=stats_path,
			cuda=args.cuda,
			num_workers=args.fid_num_workers,
			gpu_ids=gpu_ids if gpu_ids else None,
		)


if __name__ == "__main__":
	main()
