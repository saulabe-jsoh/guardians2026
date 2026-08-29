
# 가디언즈(Gardians) 실시간 위험도 모니터링 시스템

이 프로젝트는 112 신고 접수 내용 및 대화 텍스트를 기반으로 **스토킹 등 범죄 위험도를 실시간으로 분류하는 AI 시스템**입니다. 
Scikit-learn 기반의 Logistic Regression(TF-IDF) 모델을 학습하여 ONNX 형식으로 변환한 뒤, 파이썬 로컬 환경 또는 웹 브라우저 상에서 서버 없이 직접 추론을 수행합니다.

---

## 📂 프로젝트 구조

```text
├── README.md                                 # 프로젝트 설명서 (현재 파일)
├── requirements.txt                          # 파이썬 의존성 패키지 목록
├── run.bat                                   # 로컬 테스트 스크립트 실행 배치 파일
├── train_gardians_model.py                   # (선택) ONNX 모델 학습 스크립트
├── test_gardians_model_local_2.py            # 로컬 110건 추론 및 테스트 스크립트
├── index.html                                # 웹 브라우저 단독 실행용 모니터링 대시보드
├── gardians_model.onnx                       # 훈련된 위험도 분류 AI 모델
└── data/                                     # 학습 데이터 (CSV, XLSX 등)

# 실행 방법

Bash 

run.bat


🚀 1. 설치 및 환경 설정 (Requirements)파이썬 3.8 이상의 환경이 필요합니다. 

제공된 requirements.txt를 사용하여 필수 패키지를 설치하십시오.

설치 명령어:

Bash

pip install -r requirements.txt

🏃 2. 로컬 테스트 실행 (test_gardians_model_local_2.py)

학습된 gardians_model.onnx 모델이 정상적으로 작동하는지 확인하는 과정입니다.

실행 방법 1: 배치 파일 사용 (run.bat)윈도우 환경에서는 프로젝트 폴더 내의 run.bat 파일을 더블클릭하여 바로 테스트를 실행할 수 있습니다. 

스크립트가 종료된 후 창이 바로 닫히지 않고 결과를 확인할 수 있도록 pause 명령어가 포함되어 있습니다.

실행 방법 2: 파이썬 직접 실행명령 프롬프트(CMD) 또는 터미널을 열고 직접 스크립트를 실행합니다.

Bash

python test_gardians_model_local_2.py

동작 원리

110건의 사전 정의된 샘플 문장(저위험, 중위험, 고위험)을 모델에 입력합니다. 

ONNX 분류 모델이 뱉어내는 각 클래스의 확률 배열을 [0.1, 0.6, 0.9] 등의 가중치와 곱하여 0.0 ~ 1.0 사이의 연속적인 위험도 점수(risk_score)를 계산 및 출력합니다.  

🌐 3. 웹 브라우저 대시보드 (index.html)

파이썬 백엔드 서버 없이 HTML과 JavaScript만으로 모델을 구동하는 스탠드얼론 대시보드입니다. 

onnxruntime-web 라이브러리를 사용하여 브라우저 내부에서 실시간 추론을 수행합니다.

실행 방법

GitHub Pages (웹 호스팅 모드): https://saulabe-jsoh.github.io/guardians2026/release/test-gt.html

프로젝트 폴더를 GitHub Pages로 배포할 때, index.html과 gardians_model.onnx 파일을 같은 경로에 둡니다. 

웹 접속 시 모델을 자동 탐색하여 로드합니다.로컬 파일 모드:내 컴퓨터에서 index.html을 브라우저로 바로 엽니다 (더블클릭).

상단의 [💻 로컬 파일 모드]를 클릭하고 [파일 선택] 버튼을 눌러 gardians_model.onnx 파일을 직접 선택합니다.

주요 기능반원형 스피도미터 계기판: 실시간 위험도에 따라 초록(안전) >> 노랑(경계) >> 빨강(긴급) 3단계로 바늘이 회전합니다.

장문 시나리오 

자동 재생: 5가지 스토킹 상황(각 50문장)을 순차적으로 자동 재생하며 실시간 위기 고조 상황을 시뮬레이션합니다.

누적 추이 차트: Chart.js를 연동하여 입력된 사건별 누적 위험도 시계열 흐름을 꺾은선으로 가시화합니다.

🧠 4. 모델 재학습 (Optional)※ 이미 학습된 gardians_model.onnx 파일이 존재할 경우 생략 가능합니다.

모델을 처음부터 다시 학습하려면, 학습 데이터가 존재하는지 확인한 후 학습 스크립트를 실행합니다.

CRITICAL_SOS_SAMPLES 데이터를 증강하여 초단기 긴급 문장("살려주세요!" 등)을 고위험으로 안정적으로 잡아낼 수 있도록 학습 파이프라인이 보정되어 있습니다.
