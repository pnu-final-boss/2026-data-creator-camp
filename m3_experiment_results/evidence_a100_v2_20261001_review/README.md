# Mission 3 저장 결과·중단 원인 분석

2026-10-01. `report.html`을 내려받아 브라우저로 여세요.

- 신규 학습/추론 없이 저장 결과만 분석했습니다. 작업은 중단 상태이며 GPU 재실행하지 않았습니다.
- Qwen 896통 × 3정책 완료, EXAONE8통 부분, 나머지 모델 미실행입니다.
- dev384통에서 direct Macro-F1 0.56947, direct+redacted 0.59862입니다.
- ReAct는 최종 재판정 27/384통만 완료해 정상 ReAct 성능으로 해석할 수 없습니다.
- 최고 결합은 주로 FP 감소 효과이며, 고정 라벨 빈도 대조군보다 추가 우위는 아직 입증되지 않았습니다.
- 점수 비교와 confidence interval은 개발 표본에서 탐색적으로 산출했습니다.

파일: report.html(설명), analysis.json(대응 비교), statistics.json(추가 검정/라벨 지표),
diagnosis.json(형식·행동 실패 집계), reproducibility.json(재현 정보), leaderboard.csv(점수표).
전사, 개별 ID, 원래 오류 응답, 가중치, 캐시, 서버 인증정보는 포함하지 않습니다.
