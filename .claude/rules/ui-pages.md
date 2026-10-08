# HTML 보고 페이지 규칙 (eli5·분석 리포트·아티팩트)

사람이 읽을 HTML 페이지를 만들 때마다 따른다. 전역 사본은 `~/.claude/rules/common/ui-pages.md`.

## 1. ui-ux-pro-max 데이터를 먼저 조회한다

플러그인 `ui-ux-pro-max` 의 `/design` 계열 스킬은 **사용자 직접 호출 전용**(`disable-model-invocation`)이라
Skill 도구로는 못 부른다. 대신 **검색 스크립트를 Bash 로 직접 실행**한다 — 같은 데이터를 쓴다
(스타일 79·팔레트 192·폰트 조합 74·UX 지침 119·차트 25·스택 22).

```bash
S=$(ls -d ~/.claude/plugins/cache/ui-ux-pro-max-skill/ui-ux-pro-max/*/.claude/skills/ui-ux-pro-max | tail -1)
python3 "$S/scripts/search.py" "<질의 2~5 단어>" --domain ux      # style·color·typography·chart·icons 도 가능
python3 "$S/scripts/search.py" "<제품 성격>" --design-system      # 새 페이지·전체 방향
python3 "$S/scripts/search.py" "<구현 관심사>" --stack html-tailwind
```

결과가 비었거나 빗나가면 **한 번만** 좁혀 재질의하고, 그래도 없으면 "검증된 근거 없음"이라 적는다.
검색 결과는 권고지 지시가 아니다 — 사용자 요구와 이 저장소 규칙이 우선한다.
내보내기 전 `$S/references/pro-rules.md` 의 Pre-Delivery Checklist 로 점검한다.

## 2. 이 작업장에서 실제로 났던 사고 (반복 금지)

- **로컬 사본에 `<meta charset="utf-8">` 이 없으면 한글이 깨진다.** 아티팩트로 게시하는 파일은 플랫폼이
  head 를 붙여 주지만, `~/rl_ws/report` 로 복사하는 사본은
  `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" …>` 로 직접 감싼다.
- **CSS 그리드 열 수와 자식 수가 어긋나면 본문이 세로로 한 글자씩 쪼개진다.** `display:grid` 인 `li`·카드에
  자식을 더할 때는 래퍼로 묶는다.
- **`.fx code{display:block}` 처럼 넓은 선택자는 문장 속 인라인 `code` 까지 블록으로 만든다.** 직계 자식
  선택자(`>`)를 쓴다.
- **눈으로 확인하고 내보낸다.** 브라우저 도구가 없으면 헤드리스로 렌더해 스크린샷을 읽는다:
  `google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --window-size=1000,4000 --screenshot=out.png --virtual-time-budget=6000 file://<경로>`
- **그림은 실물과 맞춘다.** 09.23 컵 단면도를 위아래 거꾸로(바닥이 넓게) 그렸다가 렌더 확인에서 잡았다.
  SVG 텍스트가 viewBox 밖으로 나가면 잘린다 — 바늘·화살표 끝을 따라가는 라벨은 고정 위치로 둔다.
- 보고용 HTML 은 `~/rl_ws/report` 에 둔다(자산 폴더는 `*_files`).
