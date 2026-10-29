# ============================================================
# V6：固定训练、校准、阈值开发与BrEaST外部评估
# CPU；3组特征 × LR/集成；不搜索参数
# ============================================================

from pathlib import Path
import os
import json
import time
import hashlib
import zipfile
import shutil
import platform
import importlib.metadata

import numpy as np
import pandas as pd
import joblib

from scipy.special import logit
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits
from IPython.display import display, FileLink

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

# 多份V6结果时，填写要用的ZIP或目录。
FEATURE_PATH = ""

SEED = 42
TARGET_SENSITIVITY = 0.95
BOOTSTRAPS = 2000
THREADS = 2


# ---------- 1. 输入与安全解压 ----------
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(src, target):
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()

    with zipfile.ZipFile(src) as z:
        total = 0
        for item in z.infolist():
            p = (root / item.filename).resolve()
            total += item.file_size

            if p != root and root not in p.parents:
                raise ValueError("ZIP包含异常路径。")
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("ZIP包含符号链接。")
            if total > 2_000_000_000:
                raise ValueError("V6特征结果包异常大。")

        z.extractall(root)


def find_features():
    required = {
        "features.csv", "cohort.csv", "protocol.json", "summary.json"
    }

    def valid(p):
        return p.is_dir() and all((p / n).is_file() for n in required)

    if FEATURE_PATH.strip():
        selected = Path(FEATURE_PATH.strip())
    else:
        candidates = [
            p for p in WORK.glob("bus_v6_features_*")
            if valid(p)
        ]

        if not candidates:
            candidates = [
                p.parent for p in INPUT.rglob("features.csv")
                if valid(p.parent)
                and (p.parent / "v6_extractor.py").is_file()
            ]

        candidates = sorted(set(candidates))

        if not candidates:
            candidates = sorted(INPUT.rglob("bus_v6_features_*.zip"))

        if len(candidates) != 1:
            print("找到的候选：")
            for p in candidates:
                print(p)
            raise RuntimeError(
                "请在FEATURE_PATH填写唯一的V6结果ZIP或目录。"
            )

        selected = candidates[0]

    if selected.is_file():
        if selected.suffix.lower() != ".zip":
            raise ValueError("FEATURE_PATH应为ZIP或结果目录。")

        target = WORK / "v6_train_cache" / sha_file(selected)[:16]
        safe_extract(selected, target)

        found = [
            p.parent for p in target.rglob("features.csv")
            if valid(p.parent)
        ]
        if len(found) != 1:
            raise ValueError("ZIP内的V6结果目录不唯一。")

        selected = found[0]

    if not valid(selected):
        raise ValueError("所选目录缺少V6必要文件。")

    return selected


# ---------- 2. 固定模型 ----------
def make_models():
    def prep(model, scale=False):
        steps = [
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )
        ]
        if scale:
            steps.append(StandardScaler())
        return make_pipeline(*steps, model)

    return {
        "LR": prep(
            LogisticRegression(
                C=1,
                max_iter=3000,
                solver="lbfgs",
                random_state=SEED,
            ),
            scale=True,
        ),
        "ET": prep(
            ExtraTreesClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=SEED,
                n_jobs=THREADS,
            )
        ),
        "HGB": prep(
            HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=1,
                early_stopping=False,
                random_state=SEED,
            )
        ),
    }


def patient_probability(pipeline, frame, columns):
    values = pipeline.predict_proba(frame[columns])[:, 1]
    return (
        pd.Series(values, index=frame.patient_key.to_numpy())
        .groupby(level=0)
        .mean()
        .sort_index()
    )


def calibrated_probability(raw, calibrator):
    x = logit(
        np.clip(raw.to_numpy(), 1e-6, 1 - 1e-6)
    ).reshape(-1, 1)

    return pd.Series(
        calibrator.predict_proba(x)[:, 1],
        index=raw.index,
    )


def fast_auc(y, score):
    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    ranks = rankdata(score, method="average")
    return float(
        (ranks[y == 1].sum() - n1 * (n1 + 1) / 2)
        / (n1 * n0)
    )


def wilson(successes, total):
    if not total:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = (
        z * np.sqrt(
            p * (1 - p) / total + z * z / (4 * total * total)
        ) / den
    )
    return float(center - radius), float(center + radius)


def main_train():
    src = find_features()
    print("使用特征结果：", src, flush=True)

    feature_protocol = json.loads(
        (src / "protocol.json").read_text()
    )
    summary = json.loads((src / "summary.json").read_text())

    if feature_protocol.get("version") != "V6-unified-features-1":
        raise ValueError("此训练代码只接受指定V6特征版本。")

    if (
        summary.get("errors") != 0
        or summary.get("processed_images") != 2127
        or summary.get("exact_cross_dataset_overlap_rows") != 0
    ):
        raise ValueError("V6提取或重复检查未通过，先处理问题。")

    extractor = src / "v6_extractor.py"
    if not extractor.is_file():
        raise FileNotFoundError("缺少V6提取源码。")

    if sha_file(extractor) != feature_protocol["extractor_sha256"]:
        raise ValueError("提取源码与协议哈希不符。")

    meta = pd.read_csv(
        src / "cohort.csv",
        dtype={
            "uid": str,
            "image_id": str,
            "patient_id": str,
            "patient_key": str,
        },
    )
    features = pd.read_csv(src / "features.csv", dtype={"uid": str})

    if (
        not meta.uid.is_unique
        or not features.uid.is_unique
        or set(meta.uid) != set(features.uid)
    ):
        raise ValueError("特征和清单ID不匹配。")

    data = meta.merge(features, on="uid", validate="one_to_one")
    data["label"] = pd.to_numeric(data.label, errors="raise").astype(int)

    if set(data.label) != {0, 1}:
        raise ValueError("诊断标签必须为0/1。")

    if not data.groupby("patient_key").label.nunique().eq(1).all():
        raise ValueError("同一患者存在冲突诊断标签。")

    base = list(feature_protocol["feature_sets"]["base"])
    sets = {
        "base": base,
        "base_missing": base + ["postdepth__missing"],
        "base_posterior": base + [
            "postdepth__missing", "postdepth__delta"
        ],
    }

    all_columns = list(dict.fromkeys(
        c for columns in sets.values() for c in columns
    ))

    for columns in sets.values():
        if len(columns) != len(set(columns)):
            raise ValueError("特征列表存在重复列。")

    for col in all_columns:
        data[col] = pd.to_numeric(data[col], errors="raise")

    if np.isinf(data[all_columns].to_numpy()).any():
        raise ValueError("特征中存在无穷值。")

    if not data["postdepth__missing"].isin([0, 1]).all():
        raise ValueError("后方缺失标记不是0/1。")

    if not np.array_equal(
        data["postdepth__delta"].isna().to_numpy(),
        data["postdepth__missing"].eq(1).to_numpy(),
    ):
        raise ValueError("后方测量缺失值与标记不一致。")

    bus = data[data.dataset == "BUS-BRA"].copy()
    ext = data[data.dataset == "BrEaST"].copy()

    if (
        len(bus) != 1875
        or bus.patient_key.nunique() != 1064
        or len(ext) != 252
        or ext.patient_key.nunique() != 252
    ):
        raise ValueError("数据量不符合预期。")

    # 标签不进入特征列表。
    prohibited = {
        "label", "patient_id", "patient_key", "uid",
        "Classification", "Posterior_features",
    }
    if prohibited & set(all_columns):
        raise ValueError("预测特征中混入标签或ID。")

    source_y = bus.groupby("patient_key").label.first().sort_index()

    # 固定一次患者级分层划分。大约60%/20%/20%。
    train_ids, remaining_ids = train_test_split(
        source_y.index.to_numpy(),
        test_size=0.40,
        stratify=source_y.to_numpy(),
        random_state=SEED,
    )
    cal_ids, gate_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        stratify=source_y.loc[remaining_ids].to_numpy(),
        random_state=SEED + 1,
    )

    roles = {
        "train": sorted(train_ids),
        "calibration": sorted(cal_ids),
        "threshold": sorted(gate_ids),
    }

    role_sets = [set(v) for v in roles.values()]
    if (
        set.union(*role_sets) != set(source_y.index)
        or sum(map(len, role_sets)) != len(source_y)
    ):
        raise ValueError("来源数据划分有重叠或遗漏。")

    frames = {
        role: bus[bus.patient_key.isin(ids)].copy()
        for role, ids in roles.items()
    }

    train = frames["train"]

    # 每位患者总训练权重相同，再使两类总权重相同。
    image_counts = train.groupby("patient_key").size()
    class_counts = source_y.loc[roles["train"]].value_counts()

    weights = (
        1.0 / train.patient_key.map(image_counts).to_numpy()
        / train.label.map(class_counts).to_numpy()
    )
    weights /= weights.mean()

    run = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()%1000000}"
    out = WORK / ("bus_v6_external_training_" + run)
    out.mkdir(parents=True)
    (out / "models").mkdir()

    split_table = pd.DataFrame([
        {
            "patient_key": patient,
            "role": role,
            "label": int(source_y.loc[patient]),
        }
        for role, ids in roles.items()
        for patient in ids
    ])
    split_table.to_csv(out / "source_patient_split.csv", index=False)

    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]:
        versions[package] = importlib.metadata.version(package)

    protocol = {
        "version": "V6-external-fixed-1",
        "seed": SEED,
        "feature_sets": sets,
        "source_split": "Patient-stratified 60% train, 20% calibration, 20% threshold",
        "imputation": "Training-only median, keep_empty_features=True",
        "scaling": "Training-only StandardScaler for LR",
        "weighting": "Equal patient weight, equal class total weight",
        "ensemble": "Mean LR/ET/HGB patient probabilities, then separate Platt calibration",
        "calibration": "Unweighted LR(C=1000) on clipped logit of patient probabilities",
        "threshold": "Highest threshold attaining >=95% empirical sensitivity in source threshold subset",
        "target_sensitivity": TARGET_SENSITIVITY,
        "external_recalibration": False,
        "parameter_search": False,
        "primary_model": "LR",
        "primary_increment": "base_posterior minus base_missing",
        "secondary_model": "ensemble",
        "bootstrap_replicates": BOOTSTRAPS,
        "ci_note": "Conditional on fitted models; does not include training-split uncertainty",
        "multiplicity": "Secondary intervals are descriptive and unadjusted",
        "prior_external_use": (
            "BrEaST posterior-sign labels were previously examined in V5B; "
            "disclose this development history"
        ),
        "feature_protocol": feature_protocol,
        "input_hashes": {
            name: sha_file(src / name)
            for name in ["features.csv", "cohort.csv", "v6_extractor.py"]
        },
        "versions": versions,
        "model_parameters": {
            name: {
                k: str(v) if not isinstance(
                    v, (str, int, float, bool, type(None))
                ) else v
                for k, v in model.get_params().items()
            }
            for name, model in make_models().items()
        },
    }

    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(extractor, out / "v6_extractor.py")

    try:
        cell = get_ipython().history_manager.input_hist_raw[-1]
        (out / "training_cell.py").write_text(cell, encoding="utf-8")
    except Exception:
        pass

    print("\n患者划分：")
    print(split_table.groupby(["role", "label"]).size().unstack(fill_value=0))
    print("\n开始训练，共9次基础模型拟合。", flush=True)

    external_predictions = []
    source_predictions = []
    thresholds = []
    started = time.time()

    with threadpool_limits(limits=THREADS):
        for set_name, columns in sets.items():
            models = make_models()

            raw = {
                "calibration": {},
                "threshold": {},
                "external": {},
            }

            for model_name, pipeline in models.items():
                print(f"训练 {set_name} / {model_name}", flush=True)
                final_step = pipeline.steps[-1][0]

                pipeline.fit(
                    train[columns],
                    train.label.to_numpy(),
                    **{final_step + "__sample_weight": weights},
                )

                for role in ["calibration", "threshold"]:
                    raw[role][model_name] = patient_probability(
                        pipeline, frames[role], columns
                    )

                # 外部标签不用于预测、拟合或校准。
                raw["external"][model_name] = patient_probability(
                    pipeline, ext, columns
                )

            for role in raw:
                raw[role]["ensemble"] = pd.concat(
                    [raw[role][name] for name in ["LR", "ET", "HGB"]],
                    axis=1,
                ).mean(axis=1)

            for model_name in ["LR", "ensemble"]:
                cal_raw = raw["calibration"][model_name]

                calibrator = LogisticRegression(
                    C=1000,
                    solver="lbfgs",
                    max_iter=2000,
                )
                calibrator.fit(
                    logit(
                        np.clip(cal_raw.to_numpy(), 1e-6, 1 - 1e-6)
                    ).reshape(-1, 1),
                    source_y.loc[cal_raw.index].to_numpy(),
                )

                gate = calibrated_probability(
                    raw["threshold"][model_name], calibrator
                )
                external = calibrated_probability(
                    raw["external"][model_name], calibrator
                )

                positives = np.sort(
                    gate.loc[
                        source_y.loc[gate.index].eq(1)
                    ].to_numpy()
                )
                if len(positives) == 0:
                    raise ValueError("阈值开发集没有恶性患者。")

                allowed_misses = int(np.floor(
                    (1 - TARGET_SENSITIVITY) * len(positives) + 1e-9
                ))
                threshold = float(positives[allowed_misses])

                thresholds.append({
                    "feature_set": set_name,
                    "model": model_name,
                    "threshold": threshold,
                    "threshold_patients": len(gate),
                    "threshold_malignant": len(positives),
                    "development_sensitivity": float(
                        (positives >= threshold).mean()
                    ),
                    "calibration_slope": float(calibrator.coef_[0, 0]),
                    "calibration_intercept": float(calibrator.intercept_[0]),
                })

                external_predictions.append(pd.DataFrame({
                    "patient_key": external.index,
                    "feature_set": set_name,
                    "model": model_name,
                    "raw_p": raw["external"][model_name].loc[
                        external.index
                    ].to_numpy(),
                    "p": external.to_numpy(),
                    "threshold": threshold,
                    "prediction": (
                        external.to_numpy() >= threshold
                    ).astype(int),
                }))

                source_predictions.append(pd.DataFrame({
                    "patient_key": gate.index,
                    "feature_set": set_name,
                    "model": model_name,
                    "role": "threshold_development",
                    "label": source_y.loc[gate.index].to_numpy(),
                    "p": gate.to_numpy(),
                    "threshold": threshold,
                    "prediction": (
                        gate.to_numpy() >= threshold
                    ).astype(int),
                }))

                used_models = (
                    {"LR": models["LR"]}
                    if model_name == "LR" else models
                )
                joblib.dump(
                    {
                        "feature_columns": columns,
                        "models": used_models,
                        "model_type": model_name,
                        "calibrator": calibrator,
                        "threshold": threshold,
                        "patient_aggregation": "mean",
                        "feature_version": "V6-unified-features-1",
                    },
                    out / "models" / f"{set_name}_{model_name}.joblib",
                    compress=3,
                )

            pd.DataFrame(thresholds).to_csv(
                out / "threshold_audit.csv", index=False
            )
            print(f"完成 {set_name}", flush=True)

    # 先保存全部固定预测及模型，再合入外部结局计算成绩。
    pred = pd.concat(external_predictions, ignore_index=True)
    pred.to_csv(out / "external_predictions_frozen.csv", index=False)
    pd.concat(source_predictions, ignore_index=True).to_csv(
        out / "source_threshold_predictions.csv", index=False
    )

    model_hashes = {
        p.name: sha_file(p)
        for p in sorted((out / "models").glob("*.joblib"))
    }
    (out / "model_hashes.json").write_text(
        json.dumps(model_hashes, indent=2), encoding="utf-8"
    )

    external_y = ext.groupby("patient_key").label.first().sort_index()
    pred["label"] = pred.patient_key.map(external_y)

    if pred.label.isna().any():
        raise ValueError("外部预测与结局ID不匹配。")

    pred["label"] = pred.label.astype(int)
    pred.to_csv(out / "external_predictions.csv", index=False)

    # 所有模型使用同一批病例级Bootstrap抽样索引，便于配对比较。
    ids = external_y.index
    y = external_y.to_numpy().astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    rng = np.random.default_rng(SEED + 100)
    bootstrap_indices = [
        np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        for _ in range(BOOTSTRAPS)
    ]

    metric_rows = []
    auc_boot = {}
    probability_arrays = {}

    print("\n模型已固定，开始外部评估及置信区间计算……", flush=True)

    for (set_name, model_name), block in pred.groupby(
        ["feature_set", "model"]
    ):
        if not block.patient_key.is_unique:
            raise ValueError("同一配置的外部患者预测重复。")

        block = block.set_index("patient_key").loc[ids]
        p = block.p.to_numpy()
        d = block.prediction.to_numpy().astype(int)

        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("外部概率异常。")

        tp = int(((d == 1) & (y == 1)).sum())
        fn = int(((d == 0) & (y == 1)).sum())
        tn = int(((d == 0) & (y == 0)).sum())
        fp = int(((d == 1) & (y == 0)).sum())

        sens_ci = wilson(tp, tp + fn)
        spec_ci = wilson(tn, tn + fp)

        draws = np.array([
            fast_auc(y[idx], p[idx]) for idx in bootstrap_indices
        ])

        key = (set_name, model_name)
        auc_boot[key] = draws
        probability_arrays[key] = p

        metric_rows.append({
            "feature_set": set_name,
            "model": model_name,
            "n": len(y),
            "malignant": int(y.sum()),
            "auc": float(roc_auc_score(y, p)),
            "auc_ci_low": float(np.percentile(draws, 2.5)),
            "auc_ci_high": float(np.percentile(draws, 97.5)),
            "brier": float(brier_score_loss(y, p)),
            "sensitivity": tp / (tp + fn),
            "sensitivity_ci_low": sens_ci[0],
            "sensitivity_ci_high": sens_ci[1],
            "specificity": tn / (tn + fp),
            "specificity_ci_low": spec_ci[0],
            "specificity_ci_high": spec_ci[1],
            "missed_malignant": fn,
            "false_positive": fp,
            "threshold": float(block.threshold.iloc[0]),
        })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "external_metrics.csv", index=False)

    # 配对比较：正值表示A组AUC更高。
    paired_rows = []

    comparisons = [
        ("base_posterior", "base_missing", "measurement_increment"),
        ("base_missing", "base", "missingness_increment"),
        ("base_posterior", "base", "total_increment"),
    ]

    for model_name in ["LR", "ensemble"]:
        for a, b, meaning in comparisons:
            difference = (
                fast_auc(y, probability_arrays[(a, model_name)])
                - fast_auc(y, probability_arrays[(b, model_name)])
            )
            draws = (
                auc_boot[(a, model_name)] - auc_boot[(b, model_name)]
            )
            paired_rows.append({
                "model": model_name,
                "comparison": meaning,
                "a": a,
                "b": b,
                "auc_difference_a_minus_b": difference,
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
                "primary": bool(
                    model_name == "LR"
                    and meaning == "measurement_increment"
                ),
            })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_auc_differences.csv", index=False)

    # 校准曲线的固定等宽分箱，只描述，不据此重新校准。
    calibration_rows = []
    for (set_name, model_name), block in pred.groupby(
        ["feature_set", "model"]
    ):
        bins = np.minimum((block.p.to_numpy() * 10).astype(int), 9)
        for bin_id in range(10):
            b = block.iloc[np.flatnonzero(bins == bin_id)]
            if len(b):
                calibration_rows.append({
                    "feature_set": set_name,
                    "model": model_name,
                    "bin": bin_id,
                    "n": len(b),
                    "mean_probability": float(b.p.mean()),
                    "observed_malignant_fraction": float(b.label.mean()),
                })

    pd.DataFrame(calibration_rows).to_csv(
        out / "external_calibration_bins.csv", index=False
    )

    (out / "run_summary.json").write_text(
        json.dumps({
            "source_patients": int(len(source_y)),
            "external_patients": int(len(external_y)),
            "comparisons": len(metrics),
            "seconds": round(time.time() - started, 1),
            "completed": True,
            "note": (
                "External sensitivity is evaluated, not guaranteed to be 95%. "
                "Source threshold-set performance is developmental, not a test result."
            ),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive = Path(shutil.make_archive(str(out), "zip", out))

    print("\n外部评估：")
    display(metrics.round(4))

    print("\n配对AUC增量：")
    display(paired.round(4))

    print("\n完成。模型、阈值、患者划分、预测和源码均已保存。")
    print("结果包：", archive)
    print("请下载ZIP发给我。不要按外部成绩重新挑种子或阈值。")
    display(FileLink(str(archive)))


main_train()

# ======================== 代码结束 ========================