"""Mission 3 모델 코드: 119 신고 전사 → 9개 증상 멀티라벨 분류.

구성
- 입력: 라벨링 JSON의 utterances 전사 텍스트만 사용한다 (다른 annotation 필드는 읽지 않음).
- 모델: 한국어/다국어 사전학습 인코더(klue/roberta-large, xlm-roberta-large) + 9개 sigmoid 출력.
- 앙상블: 서로 다른 설정으로 학습한 6개 모델의 확률을 평균한다.
- 판정: 증상별 임계값. 임계값은 Training 내부 dev(10%)에서만 고른다.
- 제출 형식: 6개 모델, 토크나이저, 설정, 임계값을 .pt 파일 하나에 담아 오프라인으로 추론한다.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# 평가 대상 증상 9개. 이 순서가 모델 출력 벡터의 순서다.
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
# 임계값 탐색 격자 (0.05 ~ 0.95, 0.01 간격)
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def build_transcript(document: dict) -> str:
    """utterances를 시간 순서대로 이어 한 통화의 전사를 만든다.

    각 발화 앞에 화자 표시([대원]/[신고자])를 붙인다. 이 전처리는 라벨과 무관하게
    모든 통화에 똑같이 적용된다.
    """
    parts = []
    for utterance in document.get("utterances") or []:
        tag = "[대원]" if str(utterance.get("speaker")) == "0" else "[신고자]"
        parts.append(f"{tag} {utterance.get('text', '')}")
    return " ".join(parts)


def load_calls(label_dir: Path, with_labels: bool) -> list[dict]:
    """라벨 폴더의 JSON을 파일명 순서로 읽는다.

    with_labels=True(학습)일 때만 symptom을 읽어 9개 대상 증상의 0/1 벡터로 바꾼다.
    대상 외 증상(예: 찰과상)은 규정대로 제거되고 샘플은 유지된다.
    추론 시에는 symptom을 전혀 읽지 않는다.
    """
    calls = []
    for path in sorted(Path(label_dir).rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        call = {"stem": path.stem, "file": path.name, "text": build_transcript(document)}
        if with_labels:
            symptoms = set(document.get("symptom") or [])
            call["y"] = [int(label in symptoms) for label in LABELS]
        calls.append(call)
    return calls


def split_fit_dev(calls: list[dict], dev_size: float = 0.1, data_seed: int = 42):
    """Training을 학습용(90%)과 임계값 선택용 dev(10%)로 나눈다. Validation은 쓰지 않는다."""
    order = np.random.default_rng(data_seed).permutation(len(calls))
    cut = int(len(order) * (1 - dev_size))
    return [calls[i] for i in order[:cut]], [calls[i] for i in order[cut:]]


class CallDataset(Dataset):
    """전사를 토큰화한다. 512 토큰을 넘으면 뒷부분을 자른다 (전체 통화의 약 12%)."""

    def __init__(self, calls, tokenizer, max_length):
        self.calls, self.tokenizer, self.max_length = calls, tokenizer, max_length

    def __len__(self):
        return len(self.calls)

    def __getitem__(self, index):
        call = self.calls[index]
        item = self.tokenizer(call["text"], truncation=True, max_length=self.max_length)
        item["labels"] = [float(v) for v in call.get("y", [0] * len(LABELS))]
        return item


def make_loader(calls, tokenizer, max_length, batch_size, shuffle=False):
    def collate(batch):
        labels = torch.tensor([item.pop("labels") for item in batch])
        padded = tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded

    return DataLoader(
        CallDataset(calls, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=collate,
    )


# ---------------------------------------------------------------------------
# 손실 함수
# ---------------------------------------------------------------------------
class AsymmetricLoss(torch.nn.Module):
    """ASL (Ben-Baruch et al., 2021). 쉬운 음성 샘플의 비중을 줄이는 멀티라벨 손실.

    한 통화에서 9개 중 대부분은 음성이므로, 음성 쪽에 더 큰 감쇠(gamma_neg)를 준다.
    앙상블 구성원 중 하나를 이 손실로 학습해 다른 구성원과 오류 양상을 다르게 만든다.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip

    def forward(self, logits, targets):
        positive = torch.sigmoid(logits)
        negative = (1.0 - positive + self.clip).clamp(max=1.0)
        loss = targets * torch.log(positive.clamp(min=1e-8)) + (1 - targets) * torch.log(negative.clamp(min=1e-8))
        probability = positive * targets + negative * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        return -(loss * torch.pow(1 - probability, gamma)).sum(dim=-1).mean()


def make_criterion(loss: str, fit_calls: list[dict], pos_weight_power: float, device):
    """BCE는 희소 증상에 가중치(음성 수/양성 수)^power를 준다. macro F1이 지표라 필요하다."""
    if loss == "asl":
        return AsymmetricLoss()
    positives = np.asarray([call["y"] for call in fit_calls]).sum(axis=0)
    weight = ((len(fit_calls) - positives) / np.maximum(positives, 1)) ** pos_weight_power
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32, device=device))


# ---------------------------------------------------------------------------
# 예측과 임계값
# ---------------------------------------------------------------------------
def predict_probabilities(model, loader, device) -> np.ndarray:
    """모델의 9개 증상 확률(sigmoid)을 반환한다."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            outputs.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def tune_thresholds(truth: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """증상별로 dev F1이 가장 높은 임계값을 고른다.

    macro F1은 증상별 F1의 평균이므로 증상마다 독립적으로 최적화하면 된다.
    """
    best = []
    for i in range(len(LABELS)):
        scores = [f1_score(truth[:, i], probabilities[:, i] >= t, zero_division=0) for t in THRESHOLD_GRID]
        best.append(float(THRESHOLD_GRID[int(np.argmax(scores))]))
    return np.asarray(best)


def macro_f1(truth: np.ndarray, predictions: np.ndarray) -> float:
    """대회 지표: 증상별 F1을 따로 계산한 뒤 평균 (출제문제 10쪽)."""
    return float(f1_score(truth, predictions, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# 학습 (앙상블 구성원 1개)
# ---------------------------------------------------------------------------
def train_member(
    train_calls: list[dict],
    model_name: str,
    output: Path,
    seed: int = 42,
    loss: str = "bce",
    pos_weight_power: float = 1.0,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    device: torch.device | None = None,
) -> dict:
    """모델 하나를 학습하고, dev macro F1이 가장 높은 epoch의 가중치를 저장한다.

    dev 분할은 모든 구성원이 같도록 data_seed=42로 고정한다. seed는 초기화와
    데이터 순서만 바꾼다.
    """
    device = device or torch.device("cuda")
    torch.manual_seed(seed)
    fit_calls, dev_calls = split_fit_dev(train_calls)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fit_loader = make_loader(fit_calls, tokenizer, max_length, batch_size, shuffle=True)
    dev_loader = make_loader(dev_calls, tokenizer, max_length, batch_size)
    dev_truth = np.asarray([call["y"] for call in dev_calls])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), problem_type="multi_label_classification"
    ).to(device)
    criterion = make_criterion(loss, fit_calls, pos_weight_power, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=epochs * len(fit_loader), pct_start=0.1
    )

    best, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(fit_loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch_loss = criterion(model(**batch).logits.float(), labels)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            running += batch_loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(fit_loader)} loss={running / step:.4f}", flush=True)

        probabilities = predict_probabilities(model, dev_loader, device)
        thresholds = tune_thresholds(dev_truth, probabilities)
        score = macro_f1(dev_truth, (probabilities >= thresholds).astype(int))
        history.append({"epoch": epoch, "dev_macro_f1": score})
        print(json.dumps(history[-1]), flush=True)
        if best is None or score > best["dev_macro_f1"]:
            best = {
                "model_name": model_name,
                "max_length": max_length,
                "epoch": epoch,
                "dev_macro_f1": score,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            torch.save(best, output)
    print(f"saved {output}: dev macro F1 {best['dev_macro_f1']:.4f} at epoch {best['epoch']}")
    return {**{k: v for k, v in best.items() if k != "state_dict"}, "history": history}


# ---------------------------------------------------------------------------
# 제출용 번들: 여러 모델을 .pt 하나로 묶는다
# ---------------------------------------------------------------------------
def _pretrained_files(model_name: str) -> dict[str, bytes]:
    """토크나이저와 모델 설정 파일을 바이트로 읽어 둔다. 추론 시 인터넷이 필요 없게 하기 위함."""
    with tempfile.TemporaryDirectory() as folder:
        AutoTokenizer.from_pretrained(model_name).save_pretrained(folder)
        AutoConfig.from_pretrained(
            model_name, num_labels=len(LABELS), problem_type="multi_label_classification"
        ).save_pretrained(folder)
        return {path.name: path.read_bytes() for path in Path(folder).iterdir() if path.is_file()}


def build_bundle(member_checkpoints: dict[str, Path]) -> dict:
    """학습된 구성원 체크포인트들을 하나의 번들로 묶는다. 가중치는 float16으로 줄여 저장한다."""
    members = []
    for name, path in member_checkpoints.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        members.append(
            {
                "name": name,
                "model_name": checkpoint["model_name"],
                "max_length": checkpoint["max_length"],
                "files": _pretrained_files(checkpoint["model_name"]),
                "state_dict": {k: v.half() for k, v in checkpoint["state_dict"].items()},
            }
        )
    return {"format": "mission3_ensemble_v1", "labels": LABELS, "members": members, "thresholds": None}


def load_member(member: dict, device: torch.device):
    """번들 안의 파일만으로 토크나이저와 모델을 복원한다 (네트워크 접근 없음)."""
    with tempfile.TemporaryDirectory() as folder:
        for name, content in member["files"].items():
            Path(folder, name).write_bytes(content)
        tokenizer = AutoTokenizer.from_pretrained(folder)
        config = AutoConfig.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict({k: v.float() for k, v in member["state_dict"].items()})
    return tokenizer, model.to(device).eval()


def ensemble_probabilities(bundle: dict, calls: list[dict], device: torch.device, batch_size: int = 32) -> np.ndarray:
    """구성원을 하나씩 올려 확률을 구하고 평균한다. 한 번에 한 모델만 메모리에 둔다."""
    stack = []
    for member in bundle["members"]:
        tokenizer, model = load_member(member, device)
        loader = make_loader(calls, tokenizer, member["max_length"], batch_size)
        stack.append(predict_probabilities(model, loader, device))
        print(f"  scored {member['name']}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.mean(stack, axis=0)


def to_symptom_lists(probabilities: np.ndarray, thresholds: np.ndarray) -> list[list[str]]:
    """확률을 증상 이름 목록으로 바꾼다. 임계값을 넘는 증상이 없으면 빈 목록(0개)이다."""
    return [[LABELS[i] for i in range(len(LABELS)) if row[i] >= thresholds[i]] for row in probabilities]
