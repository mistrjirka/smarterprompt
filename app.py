import os
from typing import List
from dotenv import load_dotenv
from nicegui import ui
from orchestrator import load_config, build_bundle, ReviewLoop, ProvidersBundle

# Load .env if present
load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))

cfg = load_config()  # config.yaml or example

def build_with_overrides(main_provider: str, main_model: str, main_temp: float, main_max: int, main_stop: List[str],
                         judge_provider: str, judge_model: str, judge_temp: float, judge_max: int, judge_stop: List[str]) -> ProvidersBundle:
    base = load_config()
    base["main_ai"] = {
        "provider": main_provider, "model": main_model,
        "temperature": float(main_temp), "max_tokens": int(main_max),
        "stop": list(main_stop or []),
    }
    base["judge_ai"] = {
        "provider": judge_provider, "model": judge_model,
        "temperature": float(judge_temp), "max_tokens": int(judge_max),
        "stop": list(judge_stop or []),
    }
    return build_bundle(base)

ui.dark_mode().enable()
ui.page_title('AI Verify Loop (NiceGUI)')

ui.markdown('# AI Verify Loop (NiceGUI)').classes('text-2xl font-bold')

with ui.card().classes('w-full'):
    ui.markdown('### Configuration')
    with ui.grid(columns=4).classes('w-full gap-3'):
        main_provider = ui.select(['openai', 'ollama'], value=cfg.get('main_ai', {}).get('provider', 'openai'), label='Main provider')
        main_model = ui.input(label='Main model', value=cfg.get('main_ai', {}).get('model', 'gpt-4o-mini'))
        main_temp = ui.number(label='Main temperature', value=cfg.get('main_ai', {}).get('temperature', 0.2), step=0.1, min=0, max=2)
        main_tokens = ui.number(label='Main max tokens', value=cfg.get('main_ai', {}).get('max_tokens', 2000), step=100, min=256, max=8192)
        main_stop = ui.input(label='Main stop tokens (comma-separated)', placeholder='</final>,<|end_of_reply|>')

        judge_provider = ui.select(['ollama', 'openai'], value=cfg.get('judge_ai', {}).get('provider', 'ollama'), label='Judge provider')
        judge_model = ui.input(label='Judge model', value=cfg.get('judge_ai', {}).get('model', 'llama3'))
        judge_temp = ui.number(label='Judge temperature', value=cfg.get('judge_ai', {}).get('temperature', 0.1), step=0.1, min=0, max=2)
        judge_tokens = ui.number(label='Judge max tokens', value=cfg.get('judge_ai', {}).get('max_tokens', 1500), step=100, min=256, max=8192)
        judge_stop = ui.input(label='Judge stop tokens (comma-separated)', placeholder='</s>')

with ui.card().classes('w-full'):
    ui.markdown('### Task Prompt')
    # ⬇️ odstraněn unsupported argument auto_rows; necháme jen .props('autogrow')
    user_prompt = ui.textarea(label='Describe your task', placeholder='What should the Main AI produce?')
    user_prompt.props('autogrow')
    with ui.row():
        start_btn = ui.button('Run Review Loop', color='primary')
        finalize_btn = ui.button('Finalize', color='secondary')
        export_btn = ui.button('Export Transcript', color='gray')

with ui.card().classes('w-full'):
    ui.markdown('### Your Feedback (optional)')
    user_feedback = ui.textarea(label='What should be improved?')
    user_feedback.props('autogrow')
    iterate_btn = ui.button('Refine with User Feedback', color='primary')

# Po vytvoření tlačítek je hned znepřístupni
for b in (finalize_btn, export_btn, iterate_btn):
    try:
        b.disable()  # NiceGUI 2.x
    except Exception:
        b.props('disable')  # fallback pro starší verze

with ui.card().classes('w-full'):
    ui.markdown('### Chat')
    chat = ui.chat_message().classes('h-96')

session_loop: ReviewLoop | None = None

def add(role: str, text: str, meta: dict | None = None):
    name = {'you':'You','main':'Main AI','judge':'Judge AI','orchestrator':'Orchestrator'}.get(role, role)
    chat.append({'name': name, 'text': text})
    if meta:
        chat.append({'name': 'Meta', 'text': str(meta)})

def parse_csv(s: str) -> List[str]:
    if not s: return []
    return [x.strip() for x in s.split(',') if x.strip()]

async def on_start():
    global session_loop
    chat.clear()
    try:
        bundle = build_with_overrides(
            main_provider.value, main_model.value, main_temp.value, main_tokens.value, parse_csv(main_stop.value),
            judge_provider.value, judge_model.value, judge_temp.value, judge_tokens.value, parse_csv(judge_stop.value),
        )
        session_loop = ReviewLoop(bundle)
        add('orchestrator', 'Running Main AI…')
        main_ans = await session_loop.run_main((user_prompt.value or '').strip())
        add('main', main_ans)

        add('orchestrator', 'Running Judge AI…')
        judge_json = await session_loop.run_judge()
        add('judge', str(judge_json), {'parsed': judge_json})

        add('orchestrator', 'Refining with Judge feedback…')
        refined = await session_loop.refine()
        add('main', refined, {'phase': 'refined'})

        iterate_btn.enable()
        finalize_btn.enable()
        export_btn.enable()
        ui.notify('First review cycle complete.', type='positive')
    except Exception as e:
        ui.notify(f'Error: {e}', type='negative')

async def on_iterate():
    if not session_loop:
        ui.notify('No active session.', type='warning')
        return
    try:
        fb = (user_feedback.value or '').strip()
        if not fb:
            ui.notify('Please type some feedback.', type='warning')
            return
        add('you', f'(feedback) {fb}')
        add('orchestrator', 'Judge re-evaluates latest answer…')
        judge_json = await session_loop.run_judge()
        add('judge', str(judge_json), {'parsed': judge_json})

        add('orchestrator', 'Refining with your feedback + Judge critique…')
        refined = await session_loop.refine(user_feedback.value)
        add('main', refined, {'phase': 'refined'})
        user_feedback.value = ''
    except Exception as e:
        ui.notify(f'Error: {e}', type='negative')

async def on_finalize():
    if not session_loop:
        ui.notify('No active session.', type='warning')
        return
    try:
        add('orchestrator', 'Finalizing deliverable…')
        final_text = await session_loop.finalize()
        add('main', final_text, {'phase': 'final'})
        ui.notify('Final deliverable generated.', type='positive')
    except Exception as e:
        ui.notify(f'Error: {e}', type='negative')

async def on_export():
    if not session_loop:
        ui.notify('No active session.', type='warning')
        return
    transcript = session_loop.export()
    import json
    data = json.dumps(transcript, ensure_ascii=False, indent=2)
    # NiceGUI supports passing content directly:
    ui.download(text=data, filename='transcript.json')

# Připojení handlerů (místo dekorátorů)
start_btn.on('click', on_start)
iterate_btn.on('click', on_iterate)
finalize_btn.on('click', on_finalize)
export_btn.on('click', on_export)

ui.markdown(
    '---\n*Tip: For OpenAI set `OPENAI_API_KEY` (.env). For Ollama run `ollama serve` & `ollama pull <model>`.*'
).classes('text-sm text-gray-400')

if __name__ == '__main__':
    ui.run(host=HOST, port=PORT, reload=False)
