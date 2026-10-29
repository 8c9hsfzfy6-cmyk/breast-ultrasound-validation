# ============================================================
# V6稳定性实验：只改变来源患者划分
# 固定5次：42、43、44、45、46
# 模型随机种子固定42；不搜索参数、不选择最优重复
# ============================================================

from pathlib import Path
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

# 留空自动寻找。若提示多个候选，再填ZIP或目录路径。
FEATURE_PATH = str(sorted({p.parent for p in INPUT.rglob("features.csv") if p.parent.name.startswith("bus_v6_features_")})[0])
PREVIOUS_RUN_PATH = str(sorted({p.parent for p in INPUT.rglob("external_predictions.csv") if p.parent.name.startswith("bus_v6_external_training_")})[0])

SPLIT_SEEDS = [42, 43, 44, 45, 46]
MODEL_SEED = 42
THREADS = 2
TARGET_SENSITIVITY = 0.95


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_zip(src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()

    with zipfile.ZipFile(src) as z:
        total = 0
        for info in z.infolist():
            p = (root / info.filename).resolve()
            total += info.file_size

            if p != root and root not in p.parents:
                raise ValueError("ZIP路径异常。")
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("ZIP含符号链接。")
            if total > 2_000_000_000:
                raise ValueError("结果包大小异常。")

        z.extractall(root)


def locate_package(explicit, prefix, marker, exact_name):
    if explicit.strip():
        selected = Path(explicit.strip())
    else:
        exact_dir = WORK / exact_name
        if (exact_dir / marker).is_file():
            selected = exact_dir
        else:
            exact_zips = sorted(INPUT.rglob(exact_name + ".zip"))
            if len(exact_zips) == 1:
                selected = exact_zips[0]
            else:
                directories = [
                    p for p in WORK.glob(prefix + "*")
                    if p.is_dir() and (p / marker).is_file()
                ]
                directories += [
                    p.parent for p in INPUT.rglob(marker)
                    if (p.parent / "protocol.json").is_file()
                    and p.parent.name.startswith(prefix)
                ]
                candidates = sorted(set(directories))

                if not candidates:
                    candidates = sorted(INPUT.rglob(prefix + "*.zip"))

                if len(candidates) != 1:
                    print("候选：")
                    for p in candidates:
                        print(p)
                    raise RuntimeError(
                        f"请手动填写{prefix}对应的输入路径。"
                    )
                selected = candidates[0]

    if selected.is_file():
        if selected.suffix.lower() != ".zip":
            raise ValueError("请选择ZIP或结果目录。")

        cache = WORK / "v6_stability_cache" / sha_file(selected)[:16]
        extract_zip(selected, cache)

        found = [
            p.parent for p in cache.rglob(marker)
            if (p.parent / "protocol.json").is_file()
        ]
        if len(found) != 1:
            raise ValueError("ZIP中的结果目录不唯一。")
        selected = found[0]

    if not (selected / marker).is_file():
        raise FileNotFoundError(f"缺少{marker}：{selected}")

    return selected


def fixed_models():
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
                random_state=MODEL_SEED,
            ),
            scale=True,
        ),
        "ET": prep(
            ExtraTreesClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                max_features=0.8,
                n_jobs=THREADS,
                random_state=MODEL_SEED,
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
                random_state=MODEL_SEED,
            )
        ),
    }


def patient_score(pipeline, frame, columns):
    p = pipeline.predict_proba(frame[columns])[:, 1]
    return (
        pd.Series(p, index=frame.patient_key.to_numpy())
        .groupby(level=0).mean().sort_index()
    )


def apply_calibration(raw, model):
    values = logit(
        np.clip(raw.to_numpy(), 1e-6, 1 - 1e-6)
    ).reshape(-1, 1)
    return pd.Series(
        model.predict_proba(values)[:, 1],
        index=raw.index,
    )


def main_stability():
    src = locate_package(
        FEATURE_PATH,
        "bus_v6_features_",
        "features.csv",
        "bus_v6_features_20261013_120420_94529",
    )
    previous = locate_package(
        PREVIOUS_RUN_PATH,
        "bus_v6_external_training_",
        "external_predictions.csv",
        "bus_v6_external_training_20261013_121259_911015",
    )

    print("特征目录：", src)
    print("原训练目录：", previous)

    old_protocol = json.loads(
        (previous / "protocol.json").read_text()
    )

    if old_protocol.get("version") != "V6-external-fixed-1":
        raise ValueError("原训练协议版本不匹配。")

    # 必须与上一轮使用完全相同的特征及清单。
    for name, expected in old_protocol["input_hashes"].items():
        if sha_file(src / name) != expected:
            raise ValueError(f"{name}与上一轮输入不一致。")

    sets = old_protocol["feature_sets"]
    if set(sets) != {"base", "base_missing", "base_posterior"}:
        raise ValueError("特征组不符合预期。")

    versions = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]:
        versions[name] = importlib.metadata.version(name)

    changed = {
        name: (old_protocol["versions"].get(name), version)
        for name, version in versions.items()
        if old_protocol["versions"].get(name) != version
    }

    if changed:
        print("运行环境与上一轮不同：", changed)
        print("后面将严格检查seed=42能否复现。")

    meta = pd.read_csv(
        src / "cohort.csv",
        dtype={"uid": str, "patient_key": str},
    )
    feature = pd.read_csv(src / "features.csv", dtype={"uid": str})

    if (
        not meta.uid.is_unique
        or not feature.uid.is_unique
        or set(meta.uid) != set(feature.uid)
    ):
        raise ValueError("清单与特征ID不匹配。")

    data = meta.merge(feature, on="uid", validate="one_to_one")
    data["label"] = pd.to_numeric(
        data.label, errors="raise"
    ).astype(int)

    columns_all = list(dict.fromkeys(
        col for columns in sets.values() for col in columns
    ))
    for col in columns_all:
        data[col] = pd.to_numeric(data[col], errors="raise")

    if np.isinf(data[columns_all].to_numpy()).any():
        raise ValueError("特征存在无穷值。")
    if not data.groupby("patient_key").label.nunique().eq(1).all():
        raise ValueError("患者标签冲突。")

    bus = data[data.dataset == "BUS-BRA"].copy()
    ext = data[data.dataset == "BrEaST"].copy()

    if (
        len(bus) != 1875
        or bus.patient_key.nunique() != 1064
        or len(ext) != 252
        or ext.patient_key.nunique() != 252
    ):
        raise ValueError("数据量不符合预期。")

    y_source = bus.groupby("patient_key").label.first().sort_index()
    y_external = ext.groupby("patient_key").label.first().sort_index()

    old_predictions = pd.read_csv(
        previous / "external_predictions.csv",
        dtype={"patient_key": str},
    )

    run = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()%1000000}"
    out = WORK / ("bus_v6_stability_" + run)
    out.mkdir(parents=True)
    (out / "models").mkdir()

    protocol = {
        "version": "V6-split-stability-1",
        "split_seeds": SPLIT_SEEDS,
        "model_seed": MODEL_SEED,
        "split": "Source patient-stratified 60/20/20",
        "feature_sets": sets,
        "target_development_sensitivity": TARGET_SENSITIVITY,
        "parameter_search": False,
        "best_seed_selection": False,
        "primary_comparison": "LR: base_posterior minus base_missing",
        "external_cohort": "Same 252 BrEaST cases in every repetition",
        "interpretation": (
            "Descriptive sensitivity to source splitting, "
            "not five independent external cohorts or a new sample size."
        ),
        "ci_policy": (
            "Report across-split mean, SD and range descriptively; "
            "do not use SD/sqrt(5) as an external-performance confidence interval."
        ),
        "prior_protocol_sha256": sha_file(previous / "protocol.json"),
        "prior_input_hashes": old_protocol["input_hashes"],
        "versions": versions,
    }

    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(src / "v6_extractor.py", out / "v6_extractor.py")

    try:
        cell = get_ipython().history_manager.input_hist_raw[-1]
        (out / "executed_cell.py").write_text(cell, encoding="utf-8")
    except Exception:
        pass

    all_predictions = []
    all_splits = []
    all_thresholds = []
    reproducibility = []
    started = time.time()

    with threadpool_limits(limits=THREADS):
        for split_seed in SPLIT_SEEDS:
            print(
                f"\n开始患者划分seed={split_seed}；模型seed固定42",
                flush=True,
            )

            train_ids, remaining = train_test_split(
                y_source.index.to_numpy(),
                test_size=0.40,
                stratify=y_source.to_numpy(),
                random_state=split_seed,
            )
            cal_ids, gate_ids = train_test_split(
                remaining,
                test_size=0.50,
                stratify=y_source.loc[remaining].to_numpy(),
                random_state=split_seed + 1,
            )

            roles = {
                "train": sorted(train_ids),
                "calibration": sorted(cal_ids),
                "threshold": sorted(gate_ids),
            }
            groups = [set(x) for x in roles.values()]
            if (
                set.union(*groups) != set(y_source.index)
                or sum(map(len, groups)) != len(y_source)
            ):
                raise ValueError("患者划分有重叠或遗漏。")

            for role, ids in roles.items():
                for patient in ids:
                    all_splits.append({
                        "split_seed": split_seed,
                        "patient_key": patient,
                        "role": role,
                        "label": int(y_source.loc[patient]),
                    })

            frames = {
                role: bus[bus.patient_key.isin(ids)].copy()
                for role, ids in roles.items()
            }
            train = frames["train"]

            image_counts = train.groupby("patient_key").size()
            class_counts = y_source.loc[roles["train"]].value_counts()
            weights = (
                1.0 / train.patient_key.map(image_counts).to_numpy()
                / train.label.map(class_counts).to_numpy()
            )
            weights /= weights.mean()

            seed_predictions = []

            for set_name, columns in sets.items():
                models = fixed_models()
                raw = {
                    "calibration": {},
                    "threshold": {},
                    "external": {},
                }

                for name, pipeline in models.items():
                    final_step = pipeline.steps[-1][0]
                    pipeline.fit(
                        train[columns],
                        train.label.to_numpy(),
                        **{final_step + "__sample_weight": weights},
                    )

                    for role in ["calibration", "threshold"]:
                        raw[role][name] = patient_score(
                            pipeline, frames[role], columns
                        )
                    raw["external"][name] = patient_score(
                        pipeline, ext, columns
                    )

                for role in raw:
                    raw[role]["ensemble"] = pd.concat(
                        [raw[role][n] for n in ["LR", "ET", "HGB"]],
                        axis=1,
                    ).mean(axis=1)

                calibrators = {}
                thresholds = {}

                for name in ["LR", "ensemble"]:
                    cal_raw = raw["calibration"][name]

                    platt = LogisticRegression(
                        C=1000, solver="lbfgs", max_iter=2000
                    )
                    platt.fit(
                        logit(
                            np.clip(
                                cal_raw.to_numpy(), 1e-6, 1 - 1e-6
                            )
                        ).reshape(-1, 1),
                        y_source.loc[cal_raw.index].to_numpy(),
                    )

                    gate = apply_calibration(
                        raw["threshold"][name], platt
                    )
                    test = apply_calibration(
                        raw["external"][name], platt
                    )

                    positives = np.sort(
                        gate.loc[
                            y_source.loc[gate.index].eq(1)
                        ].to_numpy()
                    )
                    allowed = int(np.floor(
                        (1 - TARGET_SENSITIVITY)
                        * len(positives) + 1e-9
                    ))
                    threshold = float(positives[allowed])

                    calibrators[name] = platt
                    thresholds[name] = threshold

                    all_thresholds.append({
                        "split_seed": split_seed,
                        "feature_set": set_name,
                        "model": name,
                        "threshold": threshold,
                        "development_malignant": len(positives),
                        "development_sensitivity": float(
                            (positives >= threshold).mean()
                        ),
                        "calibration_slope": float(platt.coef_[0, 0]),
                    })

                    seed_predictions.append(pd.DataFrame({
                        "split_seed": split_seed,
                        "patient_key": test.index,
                        "feature_set": set_name,
                        "model": name,
                        "p": test.to_numpy(),
                        "threshold": threshold,
                        "prediction": (
                            test.to_numpy() >= threshold
                        ).astype(int),
                    }))

                joblib.dump(
                    {
                        "split_seed": split_seed,
                        "model_seed": MODEL_SEED,
                        "columns": columns,
                        "models": models,
                        "calibrators": calibrators,
                        "thresholds": thresholds,
                    },
                    out / "models" / f"seed{split_seed}_{set_name}.joblib",
                    compress=3,
                )
                print(f"完成 {set_name}", flush=True)

            current = pd.concat(seed_predictions, ignore_index=True)
            current.to_csv(
                out / f"predictions_frozen_seed{split_seed}.csv",
                index=False,
            )

            # 第一轮必须复现原先seed42，避免环境或实现变化混入。
            if split_seed == 42:
                keys = ["patient_key", "feature_set", "model"]
                matched = current.merge(
                    old_predictions[keys + ["p", "threshold", "prediction"]],
                    on=keys,
                    suffixes=("_new", "_old"),
                    validate="one_to_one",
                )

                ok = (
                    len(matched) == 252 * 6
                    and np.allclose(
                        matched.p_new, matched.p_old,
                        rtol=1e-7, atol=1e-9,
                    )
                    and np.allclose(
                        matched.threshold_new, matched.threshold_old,
                        rtol=1e-7, atol=1e-9,
                    )
                    and np.array_equal(
                        matched.prediction_new, matched.prediction_old
                    )
                )

                reproducibility.append({
                    "split_seed": 42,
                    "matched_rows": len(matched),
                    "max_probability_difference": float(
                        (matched.p_new - matched.p_old).abs().max()
                    ),
                    "passed": bool(ok),
                })
                pd.DataFrame(reproducibility).to_csv(
                    out / "reproducibility_check.csv", index=False
                )

                if not ok:
                    raise RuntimeError(
                        f"seed42未复现上一轮，已停止后续重复。"
                        f"请发回{out}/reproducibility_check.csv，"
                        "不要放宽容差继续跑。"
                    )

                print("seed42复现检查通过。", flush=True)

            all_predictions.append(current)

            pd.concat(all_predictions, ignore_index=True).to_csv(
                out / "external_predictions_frozen.csv", index=False
            )
            pd.DataFrame(all_thresholds).to_csv(
                out / "threshold_audit.csv", index=False
            )
            pd.DataFrame(all_splits).to_csv(
                out / "patient_splits.csv", index=False
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions["label"] = predictions.patient_key.map(y_external)

    if predictions.label.isna().any():
        raise ValueError("外部病例标签匹配失败。")

    predictions["label"] = predictions.label.astype(int)
    predictions.to_csv(
        out / "external_predictions.csv", index=False
    )

    metric_rows = []

    for (seed, feature_set, model), b in predictions.groupby(
        ["split_seed", "feature_set", "model"]
    ):
        if len(b) != 252 or not b.patient_key.is_unique:
            raise ValueError("每个配置应有252例唯一外部病例。")

        y = b.label.to_numpy()
        p = b.p.to_numpy()
        d = b.prediction.to_numpy()

        tp = int(((y == 1) & (d == 1)).sum())
        fn = int(((y == 1) & (d == 0)).sum())
        tn = int(((y == 0) & (d == 0)).sum())
        fp = int(((y == 0) & (d == 1)).sum())

        metric_rows.append({
            "split_seed": seed,
            "feature_set": feature_set,
            "model": model,
            "n": len(b),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "missed_malignant": fn,
            "false_positive": fp,
        })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics_by_split.csv", index=False)

    descriptive = metrics.groupby(
        ["feature_set", "model"]
    )[["auc", "brier", "sensitivity", "specificity"]].agg(
        ["mean", "std", "min", "max"]
    )
    descriptive.columns = [
        f"{metric}_{stat}" for metric, stat in descriptive.columns
    ]
    descriptive.reset_index().to_csv(
        out / "descriptive_stability.csv", index=False
    )

    # 每个划分内部配对比较，不能把5次重复当作独立外部样本。
    delta_rows = []
    indexed = metrics.set_index(
        ["split_seed", "feature_set", "model"]
    )

    for seed in SPLIT_SEEDS:
        for model in ["LR", "ensemble"]:
            for a, b, name in [
                ("base_posterior", "base_missing", "measurement_increment"),
                ("base_missing", "base", "missingness_increment"),
                ("base_posterior", "base", "total_increment"),
            ]:
                aa = indexed.loc[(seed, a, model)]
                bb = indexed.loc[(seed, b, model)]
                delta_rows.append({
                    "split_seed": seed,
                    "model": model,
                    "comparison": name,
                    "auc_difference": float(aa.auc - bb.auc),
                    # 正值表示Brier降低、得到改善。
                    "brier_improvement": float(bb.brier - aa.brier),
                    "sensitivity_difference": float(
                        aa.sensitivity - bb.sensitivity
                    ),
                    "specificity_difference": float(
                        aa.specificity - bb.specificity
                    ),
                })

    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(out / "paired_differences_by_split.csv", index=False)

    delta_summary = deltas.groupby(
        ["model", "comparison"]
    ).agg(
        auc_delta_mean=("auc_difference", "mean"),
        auc_delta_min=("auc_difference", "min"),
        auc_delta_max=("auc_difference", "max"),
        positive_auc_repeats=(
            "auc_difference", lambda x: int((x > 0).sum())
        ),
        brier_improvement_mean=("brier_improvement", "mean"),
    )
    delta_summary.to_csv(out / "paired_difference_summary.csv")

    model_hashes = {
        p.name: sha_file(p)
        for p in sorted((out / "models").glob("*.joblib"))
    }
    (out / "model_hashes.json").write_text(
        json.dumps(model_hashes, indent=2), encoding="utf-8"
    )

    (out / "run_summary.json").write_text(
        json.dumps({
            "completed": True,
            "source_patients": len(y_source),
            "external_patients": len(y_external),
            "repetitions": len(SPLIT_SEEDS),
            "configurations_per_repetition": 6,
            "seed42_reproduced": True,
            "seconds": round(time.time() - started, 1),
            "no_best_seed_selected": True,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive = Path(shutil.make_archive(str(out), "zip", out))

    print("\n各配置的划分稳定性（描述性统计）：")
    display(descriptive.round(4))

    print("\n配对增量：")
    display(delta_summary.round(5))

    print("\n完成。未选择最优种子，也未调整外部阈值。")
    print("结果包：", archive)
    display(FileLink(str(archive)))


main_stability()

# ======================== 代码结束 ========================