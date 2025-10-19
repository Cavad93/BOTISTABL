# BOTISTABL-main/ml_arf.py
import os
import json
import pickle
import threading
import inspect
import math
import logging
from typing import Dict, Any

from river import ensemble, tree, forest

logger = logging.getLogger(__name__)



class SigmoidCalibrator:
    """
    Онлайн Platt scaling:
      p_cal = sigmoid(a * x + b),  где x = logit(p_raw) | p_raw

    Параметры (a, b) обучаются инкрементально (SGD) по логистической потере.
    """

    def __init__(self, lr: float = 0.05, clip: float = 10.0, use_logit: bool = True):
        self.a = 1.0
        self.b = 0.0
        self.n = 0
        self.lr = float(lr)
        self.clip = float(clip)
        self.use_logit = bool(use_logit)

    @staticmethod
    def _sigmoid(z: float) -> float:
        # численная стабильность
        if z > 709:
            return 1.0
        if z < -709:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    def predict(self, p_raw: float) -> float:
        x = self._logit(p_raw) if self.use_logit else float(p_raw)
        z = self.a * x + self.b
        z = max(min(z, self.clip), -self.clip)
        return self._sigmoid(z)

    def update(self, p_raw: float, y: int):
        x = self._logit(p_raw) if self.use_logit else float(p_raw)
        z = self.a * x + self.b
        p = self._sigmoid(z)
        e = p - int(bool(y))
        self.a -= self.lr * e * x
        self.b -= self.lr * e
        self.n += 1


class ARFModel:
    """
    Обёртка над онлайн-ансамблем river с авто-фолбэком:
      1) forest.ARFClassifier   — приоритетно (если доступен в вашей версии river)
      2) ensemble.AdaptiveRandomForestClassifier — фолбэк
      3) ensemble.StreamingRandomPatchesClassifier — если выбран/доступен
      4) ensemble.BaggingClassifier(HoeffdingTree) — финальный фолбэк

    Публичные методы:
      - predict_proba(x: Dict[str, float]) -> float
      - learn(x: Dict[str, float], y: int) -> None
      - save_now() -> None
      - is_warm(min_labels: int) -> bool
      - labels_seen() -> int
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.state_path = getattr(cfg, "ARF_STATE_PATH", "ml_state/arf_model.pkl")
        self._lock = threading.Lock()
        self._updates = 0
        self._labels_seen = 0  # сколько финальных исходов уже учтено

        # --- калибровка вероятностей ---
        self._calib = SigmoidCalibrator(
            lr=float(getattr(self.cfg, "ARF_CALIBRATION_LR", 0.05)),
            clip=float(getattr(self.cfg, "ARF_CALIBRATION_CLIP", 10.0)),
            use_logit=bool(getattr(self.cfg, "ARF_CALIBRATION_USE_LOGIT", True)),
        )
        self.calib_enabled = bool(getattr(self.cfg, "ARF_CALIBRATION_ENABLED", True))
        self.calib_min_labels = int(getattr(self.cfg, "ARF_CALIBRATION_MIN_LABELS", 50))

        self.model = self._make_model()
        self._ensure_dir()
        self._load()

    # ---------- публичные методы ----------

    def is_warm(self, min_labels: int) -> bool:
        """Готов ли использовать ML-гейтинг/калибровку по минимальному числу размеченных исходов."""
        try:
            return self._labels_seen >= int(min_labels)
        except Exception:
            return False

    def labels_seen(self) -> int:
        return int(self._labels_seen)

    def predict_proba(self, x: Dict[str, float]) -> float:
        with self._lock:
            try:
                proba = self.model.predict_proba_one(x)
                p_raw = float(proba.get(True, 0.5)) if isinstance(proba, dict) else (
                    float(proba) if proba is not None else 0.5
                )
                if self.calib_enabled and (self._labels_seen >= int(self.calib_min_labels)):
                    return float(self._calib.predict(p_raw))
                return p_raw
            except Exception:
                return 0.5

    def learn(self, x: Dict[str, float], y: int):
        with self._lock:
            # prequential: делаем прогноз до апдейта модели -> обновляем калибратор -> учим модель
            try:
                proba = self.model.predict_proba_one(x)
                p_raw = float(proba.get(True, 0.5)) if isinstance(proba, dict) else (
                    float(proba) if proba is not None else 0.5
                )
            except Exception:
                p_raw = 0.5

            if self.calib_enabled:
                try:
                    self._calib.update(p_raw, int(bool(y)))
                except Exception:
                    from error_logger import log_exception
                    log_exception("Failed to update")

            self.model.learn_one(x, bool(y))
            self._updates += 1
            self._labels_seen += 1

            save_every = int(getattr(self.cfg, "ARF_SAVE_EVERY", 50))
            if save_every > 0 and (self._updates % save_every == 0):
                self._save()

    def save_now(self):
        with self._lock:
            self._save()

    # ---------- внутреннее ----------

    def _make_model(self):
        seed = int(getattr(self.cfg, "ARF_SEED", 42))
        n_models = int(getattr(self.cfg, "ARF_N_MODELS", 15))
        lambda_value = int(getattr(self.cfg, "ARF_LAMBDA", 6))
        prefer = str(getattr(self.cfg, "ARF_ENSEMBLE", "auto")).lower()

        base_tree = tree.HoeffdingTreeClassifier(grace_period=50, delta=1e-5)

        # 1) forest.ARFClassifier — приоритетно (river>=0.20.x)
        if prefer in ("auto", "arf") and hasattr(forest, "ARFClassifier"):
            arf_init = getattr(forest, "ARFClassifier")
            try:
                sig = inspect.signature(arf_init)
                kwargs = {}
                for k, v in (("n_models", n_models), ("lambda_value", lambda_value), ("seed", seed)):
                    if k in sig.parameters:
                        kwargs[k] = v
                model = arf_init(**kwargs)
                print(f"[ARF] using forest.ARFClassifier (args={kwargs})")
                return model
            except Exception:
                from error_logger import log_exception
                log_exception("Unhandled exception")

        # 1b) фолбэк — AdaptiveRandomForestClassifier (ensemble.*)
        if prefer in ("auto", "arf") and hasattr(ensemble, "AdaptiveRandomForestClassifier"):
            try:
                model = ensemble.AdaptiveRandomForestClassifier(
                    model=base_tree,
                    n_models=n_models,
                    lambda_value=lambda_value,
                    drift_detector=None,
                    seed=seed,
                )
                print(f"[ARF] using ensemble.AdaptiveRandomForestClassifier (n={n_models}, seed={seed}) [fallback]")
                return model
            except Exception:
                from error_logger import log_exception
                log_exception("Unhandled exception")

        # 2) SRP, если явно выбран или доступен в auto
        if prefer in ("auto", "srp") and hasattr(ensemble, "StreamingRandomPatchesClassifier"):
            try:
                model = ensemble.StreamingRandomPatchesClassifier(
                    model=base_tree,
                    n_models=n_models,
                    seed=seed,
                )
                print(f"[ARF] using ensemble.StreamingRandomPatchesClassifier (n={n_models}, seed={seed})")
                return model
            except Exception:
                from error_logger import log_exception
                log_exception("Unhandled exception")

        # 3) Баггинг — финальный фолбэк
        if hasattr(ensemble, "BaggingClassifier"):
            try:
                # Сигнатуры могут отличаться между версиями river
                b_init = ensemble.BaggingClassifier
                sig = inspect.signature(b_init)
                kwargs = {}
                if "model" in sig.parameters:
                    kwargs["model"] = base_tree
                if "n_models" in sig.parameters:
                    kwargs["n_models"] = n_models
                if "seed" in sig.parameters:
                    kwargs["seed"] = seed
                model = b_init(**kwargs)
                print(f"[ARF] using ensemble.BaggingClassifier(HoeffdingTree) (args={kwargs})")
                return model
            except Exception:
                from error_logger import log_exception
                log_exception("Unhandled exception")

        raise RuntimeError("Не найден подходящий ансамбль в river.ensemble / river.forest")

    def _ensure_dir(self):
        d = os.path.dirname(self.state_path) or "."
        os.makedirs(d, exist_ok=True)

    def _save(self):
        # модель
        tmp = self.state_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.model, f)
        os.replace(tmp, self.state_path)

        # мета
        meta_path = self.state_path + ".meta.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "labels_seen": self._labels_seen,
                        "updates": self._updates,
                        "calibrator": {
                            "a": self._calib.a,
                            "b": self._calib.b,
                            "n": self._calib.n,
                            "enabled": bool(self.calib_enabled),
                            "lr": self._calib.lr,
                            "clip": self._calib.clip,
                            "use_logit": bool(self._calib.use_logit),
                            "min_labels": int(self.calib_min_labels),
                        },
                    },
                    f,
                )
        except Exception:
            from error_logger import log_exception
            log_exception("Unhandled exception")

    def _load(self):
        # модель
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path, "rb") as f:
                    self.model = pickle.load(f)
                print(f"[ARF] state loaded from {self.state_path}")
            except Exception:
                from error_logger import log_exception
                log_exception("Failed to load pickle")

        # мета
        meta_path = self.state_path + ".meta.json"
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._labels_seen = int(meta.get("labels_seen", 0))
                self._updates = int(meta.get("updates", 0))

                cal = meta.get("calibrator") or {}
                try:
                    self._calib.a = float(cal.get("a", self._calib.a))
                    self._calib.b = float(cal.get("b", self._calib.b))
                    self._calib.n = int(cal.get("n", self._calib.n))
                    self.calib_enabled = bool(cal.get("enabled", self.calib_enabled))
                    self._calib.lr = float(cal.get("lr", self._calib.lr))
                    self._calib.clip = float(cal.get("clip", self._calib.clip))
                    self._calib.use_logit = bool(cal.get("use_logit", self._calib.use_logit))
                    self.calib_min_labels = int(cal.get("min_labels", self.calib_min_labels))
                except Exception:
                    from error_logger import log_exception
                    log_exception("Unhandled exception")
            except Exception:
                from error_logger import log_exception
                log_exception("Unhandled exception")


def extract_numeric_features(d: Any, prefix: str = "") -> Dict[str, float]:
    """
    Рекурсивный сбор всех числовых фич из вложенных dict (bool исключаем).
    Пример:
        extract_numeric_features({"a": 1, "b": {"c": 2.5}}) -> {"a": 1.0, "b_c": 2.5}
    """
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            name = f"{prefix}{k}"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[name] = float(v)
            elif isinstance(v, dict):
                out.update(extract_numeric_features(v, name + "_"))
            # списки/строки/None пропускаем
    return out


# Совместимость с кодом, который по ошибке импортирует единственное число
# (в логах встречалась попытка импортировать extract_numeric_feature)
def extract_numeric_feature(d: Any, prefix: str = "") -> Dict[str, float]:
    return extract_numeric_features(d, prefix)


# ===== Фабрика для создания SHORT и LONG ARF моделей =====

def create_arf_long(cfg) -> ARFModel:
    """Создаёт ARF модель для LONG позиций"""
    return ARFModel(cfg)


def create_arf_short(cfg) -> ARFModel:
    """
    Создаёт отдельную ARF модель для SHORT позиций.
    Использует отдельный state_path, чтобы не пересекаться с LONG моделью.
    """
    # создаём копию конфига с изменённым путём для SHORT
    import copy
    cfg_short = copy.copy(cfg)
    
    # меняем путь для state SHORT модели
    original_path = getattr(cfg, "ARF_STATE_PATH", "ml_state/arf_model.pkl")
    short_path = original_path.replace(".pkl", "_short.pkl")
    cfg_short.ARF_STATE_PATH = short_path
    
    logger.info(f"[ARF SHORT] Инициализация с путём: {short_path}")
    return ARFModel(cfg_short)
