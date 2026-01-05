## Neural Photo API

Нейросетевой сервис на FastAPI для стилизации изображений с помощью Stable Diffusion XL + ControlNet и набором LoRA‑стилей (anime, cyberpunk, ghibli, noir, gogh, comix, flat_color, pointilism). Сервис принимает входное изображение в base64 и возвращает обработанный результат, а также предоставляет эндпоинты для списка стилей и проверки состояния.

### Возможности
- **Стилизация изображений**: img2img на базе SDXL + ControlNet (depth/canny).
- **LoRA‑адаптеры** по стилям, горячее переключение.
- **Автогенерация промпта** по входному изображению (BLIP captioning).
- **Настройка силы преобразования** `strength` (0–1).
- **Здоровье сервиса**: базовый и подробный статус, в т.ч. GPU и загрузка моделей.

### Требования
- Python 3.10+
- NVIDIA GPU с CUDA 12.4 (рекомендовано для производительности). В `requirements.txt` закреплены колёса PyTorch под CUDA 12.4:
  - `--index-url https://download.pytorch.org/whl/cu124`
  - `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `torchaudio==2.6.0`
- Доступ в интернет для загрузки весов с Hugging Face при первом запуске.

Можно запустить и на CPU, но будет существенно медленнее. Для CPU удалите CUDA‑специфичные строки PyTorch в `requirements.txt` и установите CPU‑сборки PyTorch с официального индекса.

### Структура
- `app/main.py` — точка входа FastAPI, CORS, исключения, роутеры, lifecycle и предзагрузка модели.
- `app/core/config.py` — настройки через Pydantic Settings.
- `app/core/security.py` — проверка API‑ключа.
- `app/routers/api_v1.py` — эндпоинты `/process`, `/styles`.
- `app/routers/health.py` — эндпоинты `/health`, `/health/status`.
- `app/services/model_loader.py` — загрузка SDXL, ControlNet, VAE, BLIP, LoRA.
- `app/services/image_processor.py` — декод/валидация изображений, маски depth/canny, инференс.
- `app/models/schemas.py` — Pydantic‑схемы запросов/ответов.
- `app/models/enums.py` — перечень доступных стилей и пути к LoRA/промптам.
- `resources/prompts/styles.json` — промпты по стилям и негативный промпт.
- `models/lora/` — директория для LoRA весов `<style>.safetensors`.

### Установка
1) Клонируйте проект и создайте окружение:
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Подготовьте ресурсы:
- Скачайте/поместите LoRA‑веса в `models/lora/` с именами:
  - `anime.safetensors`, `cyberpunk.safetensors`, `ghibli.safetensors`, `noir.safetensors`, `gogh.safetensors`, `comix.safetensors`, `flat_color.safetensors`, `pointilism.safetensors`.
- Убедитесь, что `resources/prompts/styles.json` содержит ключи для каждого стиля и общий `negative`.

3) Настройте переменные окружения (файл `.env` в корне):
```env
# Основные
API_VERSION=0.1.0
API_V1_STR=/api/v1
PROJECT_NAME=Neural Photo API
DEBUG=true

# CORS
ALLOW_ORIGINS=["*"]

# Модели (можно оставить по умолчанию для автозагрузки)
BASE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
CONTROLNET_DEPTH_MODEL=diffusers/controlnet-depth-sdxl-1.0-small
CONTROLNET_CANNY_MODEL=diffusers/controlnet-canny-sdxl-1.0-small
VAE=madebyollin/sdxl-vae-fp16-fix
PROMPT_MODEL=Salesforce/blip-image-captioning-large
DEPTH_DETECTOR=lllyasviel/ControlNet

# Производительность
ENABLE_MODEL_CPU_OFFLOAD=true
SAFETY_CHECKER=
USE_HALF_PRECISION=true

# Обработка
DEFAULT_STRENGTH=1
DEFAULT_GUIDANCE_SCALE=10
DEFAULT_STEPS=15
DEFAULT_CONTROLNET_SCALE=0.5
MAX_IMAGE_SIZE=2048

# Лимиты
RATE_LIMIT_PER_MINUTE=10

# Безопасность
API_KEY_REQUIRED=true
API_KEYS_PATH=C:/Users/<USER>/Desktop/Megamind/resources/api_keys/api_keys.txt

# Ресурсы/пути
LORA_MODELS_DIR=models/lora/
MASK_STRATEGY=depth
```

Важно:
- В `app/models/enums.py` пути к `styles.json` прописаны абсолютными путями вида `C:/Users/Anton/...`. Обновите их под свою систему или вынесите путь в переменную окружения и используйте её в коде.
- Аналогично проверьте `API_KEYS_PATH` в `config.py`.

### Запуск
Локально (перезагрузка при изменениях):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Запуск в Docker

Сервис полностью подготовлен для запуска в Docker контейнере с поддержкой GPU.

**Требования:**
- Docker Desktop (Windows) или Docker Engine (Linux).
- **NVIDIA Container Toolkit** (обязательно для Linux, в Windows входит в Docker Desktop с WSL2).

**Особенности:**
- Используется `docker-compose` для автоматического проброса GPU и настройки путей.
- Веса моделей (SDXL, ControlNet) кэшируются в Docker volume `hf_cache`, чтобы не скачивать их при каждом перезапуске.
- Пути к ключам и LoRA-моделям автоматически переопределяются внутри контейнера, править код не нужно.

**Инструкция:**

1. **Подготовьте файлы:**
   - Положите LoRA-веса в папку `models/lora/` на хосте.
   - Создайте файл `secrets/api_keys.txt` и пропишите туда ключи.
   - Убедитесь, что файлы `Dockerfile` и `docker-compose.yml` находятся в корне.

2. **Запустите контейнер:**
   ```bash
   docker compose up -d --build

Документация:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Аутентификация
- Если `API_KEY_REQUIRED=true`, необходимо использовать API‑ключ:
  - Для `POST /api/v1/process` — передать в теле `api_key`.
  - Для `GET /health/status` — передать в заголовке `api_key`.
- Файл с ключами — по пути `API_KEYS_PATH`. Каждый ключ — с новой строки.

### Эндпоинты
- `GET /health` — базовая проверка живости.
- `GET /health/status` — подробный статус (CPU, RAM, GPU, модели). Требует `api_key` в заголовке.
- `GET /api/v1/styles` — список доступных стилей.
- `POST /api/v1/process` — стилизация изображения.

Схемы:
- Request `POST /api/v1/process`:
```json
{
  "image": "<base64-строка JPEG/PNG>",
  "style": "anime|cyberpunk|ghibli|noir|gogh|comix|flat_color|pointilism",
  "strength": 0.5,
  "api_key": "<ваш_ключ>"
}
```

- Response `POST /api/v1/process`:
```json
{
  "processed_image": "<base64-строка JPEG>",
  "processing_time": 5.32,
  "style": "anime"
}
```

### Примеры
- Получить стили:
```bash
curl http://localhost:8000/api/v1/styles
```

- Подробный статус (с ключом):
```bash
curl -H "api_key: <ВАШ_КЛЮЧ>" http://localhost:8000/health/status
```

- Обработка изображения (PowerShell пример с чтением файла и base64):
```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("input.jpg"))
Invoke-RestMethod -Uri http://localhost:8000/api/v1/process -Method Post -ContentType 'application/json' -Body (
    @{ image=$b64; style='anime'; strength=0.7; api_key='<ВАШ_КЛЮЧ>' } | ConvertTo-Json
)
```

### Производительность и GPU
- При первом запуске произойдёт загрузка весов (SDXL, ControlNet, VAE, BLIP). Это может занять время и место на диске.
- Для снижения потребления VRAM включены опции `enable_model_cpu_offload` и half‑precision (`fp16`).
- Параметры инференса (`DEFAULT_STEPS`, `DEFAULT_GUIDANCE_SCALE`, `DEFAULT_CONTROLNET_SCALE`) настраиваются через `.env`.

### Тонкости путей и ресурсов
- Замените хардкоды путей в `app/models/enums.py` и `app/core/config.py` под свою среду или параметризуйте их.
- Убедитесь, что все стили из `ProcessingStyle` имеют соответствующие файлы в `models/lora/` и записи в `styles.json`.

### Лицензии и модели
Используемые модели загружаются с Hugging Face и распространяются согласно их лицензиям. Перед продакшн‑использованием проверьте совместимость лицензий.

### Запуск в продакшн (кратко)
- Используйте процесс‑менеджер (gunicorn/uvicorn workers, Windows — NSSM/Service).
- Включите кэширование моделей и прогрев на старте (`PRELOAD_BASE_MODEL=true`).
- Ограничьте CORS и храните ключи/пути в переменных окружения.

---
Если что-то не работает: проверьте версии CUDA/драйверов, корректность путей к ресурсам и наличие всех LoRA/промптов. Ошибки старта и инференса пишутся в логи приложения.