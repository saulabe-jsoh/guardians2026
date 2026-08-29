# -*- coding: utf-8 -*-
"""
저위험(경미한 갈등)·고위험(흉기·생명위협) 보강 데이터 생성기.

배경: 실측 결과 두 클래스의 학습 데이터가 부족/편향돼 있었음.
  - 저위험: peaceful_data_generated.csv 에는 "완전 무해한 일상 대화"만 있고
    "이웃 갈등·단순 시비처럼 마찰은 있지만 위협은 아닌" 사례가 없어서,
    갈등 관련 단어만 나오면 중위험으로 오분류됨.
  - 고위험: CTGAN 스토킹 데이터 중 흉기/생명위협 키워드로 골라낸 95건이
    112신고 기록 특유의 조각난 구어체라, 깔끔한 문장체(실제 데모/서비스
    입력과 같은 스타일)의 흉기·살해협박 문장을 잘 못 잡아냄.

여기서는 외부 API 없이(팀 파일에서 발견된 노출된 API 키는 사용하지 않음)
템플릿 x 슬롯 조합으로 다양한 한국어 문장을 직접 생성한다.
"""
import itertools
import os
import random

import pandas as pd

random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def wa_gwa(word):
    """받침 유무에 따라 '와'/'과' 조사를 고른다 (경비원과 O / 경비원와 X)."""
    last = word[-1]
    code = ord(last) - 0xAC00
    if 0 <= code <= 11171:
        return "와" if code % 28 == 0 else "과"
    return "와"


def with_wa(word):
    return f"{word}{wa_gwa(word)}"


def _final_consonant_code(word):
    code = ord(word[-1]) - 0xAC00
    return code % 28 if 0 <= code <= 11171 else None


def with_i_ga(word):
    fc = _final_consonant_code(word)
    return f"{word}이" if fc else f"{word}가"


def with_eul_reul(word):
    fc = _final_consonant_code(word)
    return f"{word}을" if fc else f"{word}를"

# ---------------------------------------------------------------------------
# 저위험(경미한 갈등) — 팀 test 스크립트의 Low Risk 블록(0.1~0.4)과 같은 결의 소재
# ---------------------------------------------------------------------------
low_sentences = set()

floors = ["위층", "아래층", "옆집", "윗집"]
noise_acts = ["뛰어다녀서", "의자를 끄는 소리가 나서", "세탁기를 늦은 밤에 돌려서",
              "문을 세게 닫아서", "음악을 크게 틀어서", "청소기를 새벽에 돌려서"]
for f, a in itertools.product(floors, noise_acts):
    low_sentences.add(f"{f}에서 {a} 관리사무소에 항의했습니다.")
    low_sentences.add(f"{f} 사람이 {a} 소음 갈등이 있었습니다.")

park_places = ["아파트 주차장", "골목길", "마트 주차장", "회사 주차장"]
park_others = ["이웃", "모르는 사람", "경비원", "관리인"]
for p, o in itertools.product(park_places, park_others):
    low_sentences.add(f"{p}에서 주차 문제로 {with_wa(o)} 말다툼을 했습니다.")
    low_sentences.add(f"{p}에서 {with_wa(o)} 사소한 시비가 붙었습니다.")

couple_roles = ["남편", "아내", "동거인", "친구", "형제", "부모님"]
couple_reasons = ["돈 문제", "집안일 분담", "약속 시간", "사소한 오해", "술버릇", "말투"]
for r, why in itertools.product(couple_roles, couple_reasons):
    low_sentences.add(f"{with_wa(r)} {why} 때문에 말다툼을 했습니다. 폭력은 없었습니다.")
    low_sentences.add(f"{with_wa(r)} {why}로 다퉜지만 단순 말다툼이라 특이사항은 없습니다.")

drunk_places = ["길거리", "편의점 앞", "버스정류장", "식당", "술집"]
for p in drunk_places:
    low_sentences.add(f"{p}에서 술 취한 사람과 가벼운 시비가 붙었습니다.")
    low_sentences.add(f"{p}에서 취객이 시비를 걸어 잠깐 다퉜습니다.")

ex_roles = ["헤어진 남자친구", "헤어진 여자친구", "예전 연인", "전 남친", "전 여친"]
ex_acts = ["안부 문자를 보냈", "선물을 하나 보냈", "전화를 한 통 걸었",
           "SNS에 댓글을 남겼", "잘 지내냐고 연락했"]
for r, a in itertools.product(ex_roles, ex_acts):
    low_sentences.add(f"{with_i_ga(r)} {a}을 뿐 위협적인 내용은 없었습니다.")
    low_sentences.add(f"{r}에게서 오랜만에 연락이 왔습니다. 별다른 위협은 없습니다.")

for r in ex_roles:
    low_sentences.add(f"{r}에게 안부 연락이 왔지만 위협적인 내용은 없었습니다.")

spam_topics = ["보험 가입 권유", "대출 상담", "통신사 변경 권유", "설문조사",
               "상품권 이벤트", "투자 권유"]
for t in spam_topics:
    low_sentences.add(f"모르는 번호로 {t} 전화가 계속 옵니다.")
    low_sentences.add(f"{t} 관련 스팸 전화가 하루에도 여러 번 옵니다.")

trade_roles = ["판매자", "구매자"]
trade_reasons = ["가격 문제", "환불 요청", "배송 지연", "물건 상태", "거래 취소"]
for r, why in itertools.product(trade_roles, trade_reasons):
    low_sentences.add(f"중고거래 중 {with_wa(r)} {why}로 문자로 다퉜습니다.")

work_roles = ["직장 동료", "동네 친구", "지인", "동창"]
work_reasons = ["업무 분담", "사소한 오해", "돈 문제", "약속 불이행"]
for r, why in itertools.product(work_roles, work_reasons):
    low_sentences.add(f"{with_wa(r)} {why}로 갈등이 있었지만 별다른 위협은 없었습니다.")

low_sentences.update([
    "옆집 개가 자꾸 짖어서 항의했습니다.",
    "택시기사와 요금 문제로 말다툼을 했습니다.",
    "식당에서 다른 손님과 자리 문제로 말다툼이 있었습니다.",
    "아이 층간소음 문제로 윗집과 이야기를 나눴습니다.",
    "배달 음식이 늦게 와서 업체와 실랑이를 했습니다.",
    "환불 문제로 매장 직원과 언쟁이 있었습니다.",
    "동네 주민과 쓰레기 분리수거 문제로 다퉜습니다.",
    "인터넷 설치 기사와 예약 시간 문제로 다퉜습니다.",
])

low_sentences = sorted(low_sentences)
random.shuffle(low_sentences)

# ---------------------------------------------------------------------------
# 고위험(흉기·생명위협) — 깔끔한 문장체(팀 High Risk 블록과 같은 톤)로 다양화
# ---------------------------------------------------------------------------
high_sentences = set()

weapons = ["칼", "식칼", "흉기", "가위", "둔기", "망치", "회칼"]
aggressors = ["전남친이", "전남편이", "헤어진 여자친구가", "스토킹 가해자가",
              "낯선 남자가", "모르는 사람이", "동거남이", "전여친이"]
locations = ["집 앞에서", "회사 앞에서", "골목길에서", "엘리베이터 안에서",
             "주차장에서", "현관 앞에서", "아파트 복도에서"]
fears = ["너무 무섭습니다.", "살려주세요.", "두렵습니다.", "숨어서 떨고 있습니다.",
         "빨리 와주세요.", "너무 불안합니다."]

extra_templates = [
    "{ag} 저를 차에 강제로 태워 납치하려고 했습니다. {fear}",
    "{ag} 저를 방에 가두고 감금했습니다. {fear}",
    "{ag} 집에 불을 지르겠다고 위협했습니다. {fear}",
    "{ag} 제 목을 조르며 폭행했습니다. {fear}",
    "{ag} 흉기로 위협하며 돈을 요구했습니다. {fear}",
    "{ag} 저를 인질로 삼겠다고 협박하고 있습니다. {fear}",
    "{ag} {loc} 저를 기다리다가 흉기를 꺼내들었습니다. {fear}",
    "{ag} 문을 부수고 침입해서 저를 위협했습니다. {fear}",
    "{ag} 염산을 뿌리겠다고 협박하고 있습니다. {fear}",
    "{ag} 제 아이를 납치하겠다고 전화로 협박했습니다. {fear}",
]

# 각 템플릿 그룹을 독립적으로 모은 뒤 마지막에 합쳐서, 특정 그룹(예: "칼 들고
# 죽이겠다")이 조합 수가 많다는 이유만으로 최종 500건을 독식하지 않게 한다.
# (이전 버전은 그룹별 break 기준이 뒤섞여 납치/감금/방화/인질 템플릿이
# 최종 데이터에 거의 반영되지 않는 버그가 있었음.)
kill_threat = {f"{ag} {loc} {with_eul_reul(w)} 들고 죽이겠다고 협박했습니다. {fear}"
               for w, ag, loc, fear in itertools.product(weapons, aggressors, locations, fears)}
stab_chase = {f"{ag} {w}로 저를 찌르려고 쫓아왔습니다. {fear}"
              for w, ag, loc, fear in itertools.product(weapons, aggressors, locations, fears)}
extra_variety = {tmpl.format(ag=ag, loc=loc, fear=fear)
                  for tmpl, ag, loc, fear in itertools.product(extra_templates, aggressors, locations, fears)}

high_groups = [sorted(kill_threat), sorted(stab_chase), sorted(extra_variety)]
for g in high_groups:
    random.shuffle(g)

# 그룹당 상한을 둬서 세 그룹(흉기 위협 / 추격·찌르기 / 납치·감금·방화·인질 등
# 그 외 유형)이 최종 데이터에 고르게 섞이도록 한다.
PER_GROUP_CAP = 200
high_sentences = set()
for g in high_groups:
    high_sentences.update(g[:PER_GROUP_CAP])

high_sentences = sorted(high_sentences)
random.shuffle(high_sentences)

# ---------------------------------------------------------------------------
# 중위험(미행·배회·지속적 연락) — 무기/즉각적 생명위협 없이 "스토킹 징후"만 있는
# 문장. 저위험 보강 데이터를 추가한 뒤 중위험 recall이 떨어진 것을 보완하기
# 위한 것으로, 흉기·살해 키워드는 절대 섞지 않는다(고위험과 헷갈리면 안 됨).
# ---------------------------------------------------------------------------
mid_sentences = set()

mid_aggressors = ["전남친이", "전남편이", "헤어진 여자친구가", "스토킹 가해자가",
                   "낯선 남자가", "모르는 사람이", "전여친이", "헤어진 연인이"]
mid_locations = ["집 앞에서", "회사 앞에서", "골목길에서", "집 주변에서",
                  "퇴근길에", "동네에서"]
mid_discomforts = ["불안합니다.", "스트레스가 심합니다.", "괴롭습니다.",
                    "신경이 쓰입니다.", "찝찝합니다.", "걱정됩니다."]
mid_templates = [
    "{ag} 계속 저를 미행하는 것 같습니다. {d}",
    "{ag} {loc} 자꾸 배회하고 있습니다. {d}",
    "{ag} 원치 않는 연락을 지속적으로 하고 있습니다. {d}",
    "{ag} 제 SNS를 몰래 염탐하고 있습니다. {d}",
    "{ag} 제 동선을 계속 파악하려고 합니다. {d}",
    "{ag} 거부 의사를 밝혔는데도 계속 연락합니다. {d}",
    "{ag} {loc} 서성이고 있습니다. {d}",
    "{ag} 협박성 문자를 지속적으로 보내고 있습니다. {d}",
    "{ag} 제 위치를 주변 사람들에게 캐묻고 다닙니다. {d}",
    "{ag} 수십 통씩 전화를 걸어옵니다. {d}",
    "{ag} {loc} 저를 기다리고 있습니다. {d}",
    "{ag} 만나주지 않으면 찾아오겠다고 합니다. {d}",
]
for tmpl, ag, loc, d in itertools.islice(
        itertools.product(mid_templates, mid_aggressors, mid_locations, mid_discomforts), 0, 8000):
    mid_sentences.add(tmpl.format(ag=ag, loc=loc, d=d))
    if len(mid_sentences) > 500:
        break

mid_sentences = sorted(mid_sentences)
random.shuffle(mid_sentences)

# 규모를 CTGAN 원본 대비 과도하게 부풀리지 않도록 상한선을 둔다.
low_sentences = low_sentences[:500]
high_sentences = high_sentences[:500]
mid_sentences = mid_sentences[:500]

pd.DataFrame({"학습용텍스트": low_sentences, "위험도_정답": 0.3}).to_csv(
    os.path.join(BASE_DIR, "augmented_low_risk.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"학습용텍스트": mid_sentences, "위험도_정답": 0.6}).to_csv(
    os.path.join(BASE_DIR, "augmented_mid_risk.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"학습용텍스트": high_sentences, "위험도_정답": 0.9}).to_csv(
    os.path.join(BASE_DIR, "augmented_high_risk.csv"), index=False, encoding="utf-8-sig")

print(f"저위험 보강 문장: {len(low_sentences)}건 -> augmented_low_risk.csv")
print(f"중위험 보강 문장: {len(mid_sentences)}건 -> augmented_mid_risk.csv")
print(f"고위험 보강 문장: {len(high_sentences)}건 -> augmented_high_risk.csv")
print("\n[저위험 샘플 5건]")
for s in low_sentences[:5]:
    print(" -", s)
print("\n[중위험 샘플 5건]")
for s in mid_sentences[:5]:
    print(" -", s)
print("\n[고위험 샘플 5건]")
for s in high_sentences[:5]:
    print(" -", s)
