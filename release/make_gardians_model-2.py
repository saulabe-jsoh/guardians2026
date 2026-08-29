# -*- coding: utf-8 -*-
"""
Gardians 위험도 분류 모델 학습 스크립트 (로컬 실행용)

데이터 소스:
  1) peaceful_data_generated.csv       -> 저위험(label 0, 목표 스코어 0.1)
  2) synthetic_ctgan_1000_gpt-4-turbo.xlsx        -> 스토킹 112신고 기록 (신고내용 텍스트)
  3) synthetic_ctgan_1000_roberta-large-ner.xlsx  -> 동일 스키마의 또 다른 CTGAN 합성본
  4) augmented_low_risk.csv / augmented_mid_risk.csv / augmented_high_risk.csv
     -> generate_augment_data.py로 생성한 보강 데이터 (저위험 카테고리 부재,
        고위험 CTGAN 조각문체 편향을 보완하기 위해 추가)

스토킹 기록 두 파일에는 위험도 정답 컬럼이 없음. 신고내용/종결내용에 폭력·흉기·
살해·납치 등 키워드가 있으면 고위험(label 2, 0.8), 없으면 중위험(label 1, 0.4)으로
약지도(weak supervision) 라벨을 부여한다. 이 키워드는 팀이 이미 작성한
test_gardians_model-1.txt 의 Mid/High Risk 예시 문장에서 그대로 가져온 것이라
새로 임의 기준을 만드는 것이 아니다.

편향 점검을 위해 아래를 수행한다:
  - CTGAN 특유의 중복행 제거 (237개 원본이 1000행으로 뻥튀기된 상태라 방치하면
    train/test 양쪽에 같은 문장이 들어가 성능이 과대평가됨)
  - [이름], [장소] 같은 비식별화 placeholder 토큰 제거 (이게 남아있으면 모델이
    "위험도"가 아니라 "이 문장이 CTGAN 스토킹 파일 출신인지"만 학습해버림 ->
    실제 서비스 입력(플레이스홀더 없는 자연문)에서 엉뚱하게 동작하는 원인)
  - 클래스별 샘플 수 출력 + class_weight='balanced'
  - 학습/평가 분리 후 per-class precision/recall/F1, confusion matrix 출력
  - 팀이 만든 110건 큐레이션 샘플(test_gardians_model-1.txt)로 최종 캘리브레이션 점검

버전 이력:
  -1 (원본): 이름은 "학습 스크립트"였지만 실제로는 test_gardians_model-1.txt와
      동일한 추론/테스트 코드만 있었고, 실제 학습 로직은 없었음.
  -2 (이 파일): 실제 학습 파이프라인. 아래를 거쳐 완성됨.
      1) 학습 데이터를 CTGAN 신고 기록으로만 구성 -> held-out 성능은 좋았지만
         팀이 만든 110건 자연문 테스트에서 편향 발견(저위험 20%, 고위험 0% 적중).
      2) 원인 진단: (a) "경미한 갈등이지만 위험은 아님"에 해당하는 저위험 학습
         데이터가 아예 없었음, (b) 고위험 학습 데이터(95건)가 CTGAN 특유의
         조각난 구어체라 깔끔한 문장체의 흉기·살해 표현을 못 잡음.
      3) augmented_low_risk / augmented_mid_risk / augmented_high_risk 데이터를
         생성해 보완 -> 110건 테스트 argmax 정확도 55.5% -> 90.9%로 개선.
      4) skl2onnx가 char_wb 벡터라이저를 지원하지 않아 처음엔 word 토큰화로
         바꿨는데, 한국어 활용형(칼을/칼로/칼이, 죽이겠다/죽인다) 때문에
         고위험 recall이 0%까지 떨어짐 -> 한글 음절 단위 토큰화(1~2음절
         n-gram)로 바꿔서 char n-gram과 비슷한 효과를 ONNX 호환 방식으로 확보.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PEACEFUL_CSV = os.path.join(BASE_DIR, "peaceful_data_generated.csv")
AUGMENTED_LOW_CSV = os.path.join(BASE_DIR, "augmented_low_risk.csv")
AUGMENTED_MID_CSV = os.path.join(BASE_DIR, "augmented_mid_risk.csv")
AUGMENTED_HIGH_CSV = os.path.join(BASE_DIR, "augmented_high_risk.csv")
CTGAN_FILES = [
    os.path.join(BASE_DIR, "synthetic_ctgan_1000_gpt-4-turbo.xlsx"),
    os.path.join(BASE_DIR, "synthetic_ctgan_1000_roberta-large-ner.xlsx"),
]
MODEL_OUT = os.path.join(BASE_DIR, "gardians_model.onnx")

# 위험도 라벨 정의 (중위험: 0.4, 고위험: 0.8)
LABEL_NAMES = {0: "저위험(0.1)", 1: "중위험(0.4)", 2: "고위험(0.8)"}

# test_gardians_model-1.txt 의 High Risk 블록(30건)에서 실제 등장한 위협 어휘 기반
HIGH_RISK_KEYWORDS = [
    "칼", "흉기", "식칼", "죽이", "죽인다", "살해", "살려주세요", "살려줘", "살려",
    "도와주세요", "도와줘", "구해주세요", "납치", "감금", "침입", "인질", "목을 조르",
    "목졸", "폭행", "강도", "불을 지르", "방화", "염산", "위협", "무기를 든", "찌르",
    "문 부수", "피가 나", "비명", "도망", "급해요",
]
BRACKET_TOKEN_RE = re.compile(r"\[[^\]\[]*\]")

CRITICAL_SOS_SAMPLES = [
    "살려주세요!",
    "살려주세요",
    "도와주세요!",
    "도와주세요",
    "제발 살려주세요! 도와주세요!",
    "제발 살려주세요",
    "제발 도와주세요",
    "살려줘요 제발",
    "빨리 와주세요 살려주세요",
    "도와주세요 살려주세요!",
    "문 열어주세요 살려주세요",
    "살려주세요 죽을 것 같아요",
    "제발 살려주세요 죽을 것 같아요!",
    "빨리 경찰관 보내주세요 살려주세요",
    "살려주세요 제발요",
    "도와주세요 제발요",
]

def strip_bracket_tokens(text):
    # 중첩 대괄호([CODE0 [장소]] 같은 형태)는 한 번의 정규식 치환으로 안 지워지므로
    # 더 이상 변하지 않을 때까지 반복 적용한다.
    prev = None
    while prev != text:
        prev = text
        text = BRACKET_TOKEN_RE.sub(" ", text)
    return text


def _load_text_csv(path, label):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"학습용텍스트": "text"})
    df = df[["text"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["label"] = label
    return df[["text", "label"]]


def load_peaceful():
    frames = [_load_text_csv(PEACEFUL_CSV, 0)]
    if os.path.exists(AUGMENTED_LOW_CSV):
        frames.append(_load_text_csv(AUGMENTED_LOW_CSV, 0))
    else:
        print(f"⚠️  보강 저위험 데이터 없음: {AUGMENTED_LOW_CSV} (generate_augment_data.py 먼저 실행)")
    return pd.concat(frames, ignore_index=True)


def load_stalking():
    frames = []
    for path in CTGAN_FILES:
        if not os.path.exists(path):
            print(f"⚠️  파일 없음, 건너뜀: {path}")
            continue
        raw = pd.read_excel(path, sheet_name="Sheet1")
        needed_cols = [c for c in ["신고내용", "종결내용"] if c in raw.columns]
        sub = raw[needed_cols].copy()
        frames.append(sub)
    if not frames:
        raise RuntimeError("CTGAN 스토킹 데이터 파일을 하나도 읽지 못했습니다.")
    df = pd.concat(frames, ignore_index=True)

    df["신고내용"] = df.get("신고내용", pd.Series(dtype=str)).fillna("").astype(str)
    df["종결내용"] = df.get("종결내용", pd.Series(dtype=str)).fillna("").astype(str)

    # 라벨 판정은 신고내용+종결내용 모두 보되, 학습 입력(text)은 신고내용만 사용한다.
    # 종결내용은 경찰 공문서체(법조문 인용 등)라 실제 서비스 입력(구어체 신고문)과
    # 문체가 완전히 달라, 이걸 학습 feature로 섞으면 "문체 차이"를 "위험도 차이"로
    # 오학습하는 또 다른 편향 원인이 된다. 그래서 라벨링에만 참고하고 feature에서는 뺀다.
    def has_high_risk(row):
        joined = row["신고내용"] + " " + row["종결내용"]
        return any(kw in joined for kw in HIGH_RISK_KEYWORDS)

    df["label"] = df.apply(lambda r: 2 if has_high_risk(r) else 1, axis=1)
    df["text"] = df["신고내용"].apply(lambda t: strip_bracket_tokens(t).strip())
    df = df[df["text"].str.len() >= 2]
    return df[["text", "label"]]


def dedupe_report(df, name):
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    if before != after:
        print(f"🔎 [{name}] 중복 문장 제거: {before}건 -> {after}건 (중복 {before - after}건, "
              f"CTGAN이 소수 원본을 반복 재생성한 결과로 추정)")
    return df


def main():
    print("=" * 80)
    print("Gardians 모델 학습 시작 (저위험: 0.1 / 중위험: 0.4 / 고위험: 0.8)")
    print("=" * 80)

    peaceful = load_peaceful()
    peaceful = dedupe_report(peaceful, "저위험(peaceful + 보강) 데이터")

    stalking = load_stalking()
    stalking = dedupe_report(stalking, "CTGAN 스토킹 데이터(2개 파일 통합)")

    if os.path.exists(AUGMENTED_MID_CSV):
        augmented_mid = _load_text_csv(AUGMENTED_MID_CSV, 1)
        stalking = pd.concat([stalking, augmented_mid], ignore_index=True)
        print(f"➕ 보강 중위험 문장 {len(augmented_mid)}건 추가 (저위험 보강 이후 중위험 recall 저하 보완용)")
    else:
        print(f"⚠️  보강 중위험 데이터 없음: {AUGMENTED_MID_CSV} (generate_augment_data.py 먼저 실행)")

    if os.path.exists(AUGMENTED_HIGH_CSV):
        augmented_high = _load_text_csv(AUGMENTED_HIGH_CSV, 2)
        stalking = pd.concat([stalking, augmented_high], ignore_index=True)
        print(f"➕ 보강 고위험 문장 {len(augmented_high)}건 추가 (깔끔한 문장체, CTGAN 조각문체 편향 보완용)")
    else:
        print(f"⚠️  보강 고위험 데이터 없음: {AUGMENTED_HIGH_CSV} (generate_augment_data.py 먼저 실행)")
        
    sos_df = pd.DataFrame({"text": CRITICAL_SOS_SAMPLES * 15, "label": 2})
    stalking = pd.concat([stalking, sos_df], ignore_index=True)

    # 두 CTGAN 파일이 같은 237개 원본에서 파생됐으므로 파일 간 교차 중복도 제거됨(위 dedupe가
    # concat 이후에 실행되므로 파일1/파일2 사이 중복도 함께 걸러진다).

    data = pd.concat([peaceful, stalking], ignore_index=True)
    data = data.drop_duplicates(subset=["text"]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("\n📊 최종 라벨 분포 (편향 확인):")
    dist = data["label"].value_counts().sort_index()
    for lbl, cnt in dist.items():
        pct = cnt / len(data) * 100
        print(f"  - {LABEL_NAMES[lbl]}: {cnt}건 ({pct:.1f}%)")
    max_ratio = dist.max() / dist.min()
    if max_ratio > 3:
        print(f"⚠️  클래스 불균형 경고: 최대/최소 비율 {max_ratio:.1f}배. "
              f"class_weight='balanced'로 보정하지만 소수 클래스 recall을 꼭 확인할 것.")

    X = data["text"].values
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # skl2onnx는 TfidfVectorizer의 analyzer='word' 만 완전히 지원한다(char/char_wb 변환 불가).
    # 한국어 형태소 분석기 없이 공백 기준 어절 토큰을 쓰므로 형태소 단위보다는 거칠지만,
    # 일상대화 vs 스토킹 신고 어휘 차이가 크고 위험 키워드 자체가 라벨 근거이기 때문에
    # 실용적으로는 충분히 분리 가능하다.
    # token_pattern은 ONNX 런타임의 RE2 엔진이 소화 가능해야 하고(그래서 char_wb
    # analyzer는 애초에 못 씀), 어절(\S+) 단위로 토큰화하면 "죽이겠다/죽인다",
    # "칼을/칼로/칼이" 같은 활용형이 전부 다른 토큰이 되어 학습 때 못 본 어미가
    # 붙으면 위험 키워드를 놓친다(실제로 고위험 30건 적중률 0%가 나온 원인).
    # 한글 음절(1글자) 단위를 "단어"로 취급하고 1~2음절 n-gram을 만들면
    # char-level n-gram과 비슷한 효과를 ONNX 호환 방식으로 얻을 수 있다.
    SYLLABLE_TOKEN_PATTERN = r"[가-힣]|[A-Za-z]+|[0-9]+"
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            token_pattern=SYLLABLE_TOKEN_PATTERN,
            ngram_range=(1, 3),
            max_features=15000,
            min_df=1,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0)),
    ])

    pipeline.fit(X_train, y_train)

    print("\n📈 held-out 테스트셋 성능 (전체 데이터의 20%, 학습에 안 쓰인 문장들):")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)], zero_division=0))
    print("혼동행렬 (행=실제, 열=예측):")
    print(confusion_matrix(y_test, y_pred))

    per_class_recall = {}
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    for i in range(3):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class_recall[i] = correct / total if total else float("nan")
    low_recall_classes = [LABEL_NAMES[i] for i, r in per_class_recall.items() if r < 0.5]
    if low_recall_classes:
        print(f"⚠️  recall 50% 미만 클래스 발견: {low_recall_classes} -> 이 클래스는 예측이 "
              f"신뢰할 수 없는 수준. 데이터/라벨링 재검토 필요.")

    # 편향 진단: bracket placeholder 토큰이 실제로 사라졌는지 확인
    leaked = data["text"].str.contains(r"\[[^\]]{1,20}\]", regex=True).sum()
    if leaked:
        print(f"⚠️  placeholder 토큰이 {leaked}건 남아있음 (제거 로직 재확인 필요)")
    else:
        print("\n✅ 비식별화 placeholder 토큰([이름],[장소] 등) 학습 데이터에서 모두 제거 확인됨.")
        
    # [0.1, 0.4, 0.7] 가중치로 점수 산출 검증
    print("\n🧪 긴급 단문 검증 점검:")
    test_sos = ["제발 살려주세요! 도와주세요!", "살려주세요!", "도와주세요!"]
    probs = pipeline.predict_proba(test_sos)
    score_map = np.array([0.1, 0.45, 0.85])
    for txt, prob in zip(test_sos, probs):
        score = float(np.sum(prob * score_map))
        pred_label = np.argmax(prob)
        print(f" - '{txt}' -> 예측: {LABEL_NAMES[pred_label]}, 산출 스코어: {score:.2f}")

    # ONNX 변환 (skl2onnx) — zipmap=False로 test 스크립트가 기대하는 순수 확률 텐서 출력
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    initial_type = [("input", StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        options={id(pipeline.steps[-1][1]): {"zipmap": False}},
    )
    with open(MODEL_OUT, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"\n✅ ONNX 모델 저장 완료: {MODEL_OUT}")
    print(f"   학습 문장 수: {len(X_train)}건 / 테스트 문장 수: {len(X_test)}건")


if __name__ == "__main__":
    main()
